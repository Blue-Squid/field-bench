# Host-side workflow. The Jetson copy lives at $(REMOTE_DIR) and has its own venv.
JETSON     ?= jetson
REMOTE_DIR ?= fieldbench
ARGS       ?=
CMD        ?= bench
SESSION    ?= fieldbench

.PHONY: export barcodes ocr-data barber products train-barcode sync bench live attach info modes set-mode pull report

export:            ## export ONNX models on the host
	.venv/bin/python host/export_models.py

barcodes:          ## generate the synthetic barcode dataset (needs data/coco/val2017)
	.venv/bin/python host/make_barcodes.py

ocr-data:          ## generate the synthetic OCR test + calibration frames
	.venv/bin/python host/make_text.py

barber:            ## BarBeR real photos (download first, see host/make_barber.py) -> data/barber/fieldbench
	.venv/bin/python host/make_barber.py

products:          ## Grocery Store Dataset -> data/products/{gallery,test} (product recognition)
	.venv/bin/python host/make_products.py

train-barcode:     ## fine-tune YOLO11n on it (CUDA), then export at 640/1280/1600
	.venv/bin/python host/train_barcode.py
	.venv/bin/python host/export_models.py barcode_yolo11n_640 barcode_yolo11n_1280 barcode_yolo11n_1600 --force

sync:              ## push code, scripts, ONNX and pipeline test/calibration images to the Jetson
	rsync -az --delete --exclude __pycache__ fieldbench/ $(JETSON):$(REMOTE_DIR)/fieldbench/
	rsync -az --exclude _ultralytics --exclude _ppocr models/ $(JETSON):$(REMOTE_DIR)/models/
	rsync -az scripts/ $(JETSON):$(REMOTE_DIR)/scripts/
	@if [ -d data/barcodes/test ]; then \
	  rsync -az --mkpath --relative data/barcodes/./test data/barcodes/./val/images $(JETSON):$(REMOTE_DIR)/data/barcodes/; fi
	@if [ -d data/ocr/test ]; then rsync -az --mkpath data/ocr/test data/ocr/calib $(JETSON):$(REMOTE_DIR)/data/ocr/; fi
	@if [ -d data/barber/fieldbench ]; then rsync -az --mkpath data/barber/fieldbench $(JETSON):$(REMOTE_DIR)/data/barber/; fi
	@if [ -d data/products/test ]; then rsync -az --mkpath data/products/test data/products/gallery $(JETSON):$(REMOTE_DIR)/data/products/; fi

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
