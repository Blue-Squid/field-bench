# Host-side workflow. The Jetson copy lives at $(REMOTE_DIR) and has its own venv.
JETSON     ?= jetson
REMOTE_DIR ?= fieldbench
ARGS       ?=
CMD        ?= bench
SESSION    ?= fieldbench

.PHONY: export sync bench live attach info modes set-mode pull report

export:            ## export ONNX models on the host
	.venv/bin/python host/export_models.py

sync:              ## push code and ONNX models to the Jetson
	rsync -az --delete --exclude __pycache__ fieldbench/ $(JETSON):$(REMOTE_DIR)/fieldbench/
	rsync -az --exclude _ultralytics models/ $(JETSON):$(REMOTE_DIR)/models/

info: sync
	ssh $(JETSON) 'cd $(REMOTE_DIR) && .venv/bin/python -m fieldbench info'

modes: sync        ## list nvpmodel power modes and which switch live
	ssh $(JETSON) 'cd $(REMOTE_DIR) && .venv/bin/python -m fieldbench power'

set-mode: sync     ## e.g. make set-mode MODE=7W REBOOT=1 (7W reboots the board)
	ssh $(JETSON) 'cd $(REMOTE_DIR) && .venv/bin/python -m fieldbench power --set $(MODE) $(if $(REBOOT),--reboot)'

bench: sync        ## e.g. make bench ARGS="--precisions fp16 --label quick"
	ssh -t $(JETSON) 'cd $(REMOTE_DIR) && .venv/bin/python -m fieldbench bench $(ARGS)'
	$(MAKE) pull

live: sync         ## tmux on the Jetson: fieldbench $(CMD) $(ARGS) (left) + jtop (right)
	ssh $(JETSON) 'tmux has-session -t $(SESSION) 2>/dev/null && { echo "session $(SESSION) already running: make attach"; exit 1; }; \
	  cd $(REMOTE_DIR) && tmux new-session -d -s $(SESSION) -x 230 -y 55 \
	  ".venv/bin/python -u -m fieldbench $(CMD) $(ARGS); echo; echo finished - press enter to close; read" \; \
	  split-window -h -l 42% jtop \; select-pane -L'
	@echo "attach from here:        make attach"
	@echo "attach on the Jetson:    tmux attach -t $(SESSION)"

attach:            ## watch the live session (detach with Ctrl-b d; the run keeps going)
	ssh -t $(JETSON) tmux attach -t $(SESSION)

pull:              ## fetch results JSONL back to ./results
	rsync -az $(JETSON):$(REMOTE_DIR)/results/ results/

report: pull       ## pull results and render report/index.html
	python3 host/report.py
