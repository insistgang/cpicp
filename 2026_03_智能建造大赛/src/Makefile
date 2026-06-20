# B_baseline · 一键化（容器跑训练/评测/导出）
# 用法: make build && make prepare && make train && make eval && make export
# 依赖: Docker + nvidia-container-toolkit（GPU）。Orin 端 TRT 在板上原生跑，见 trt_infer_orin.py。

IMAGE ?= searescue-b
DATA  ?= $(CURDIR)/datasets
GPU   ?= --gpus all
RUNPC  = docker run --rm $(GPU) -v $(CURDIR):/workspace -v $(DATA):/workspace/datasets $(IMAGE)

.PHONY: help build run prepare train eval export crossdomain shell clean
help:
	@echo "make build        构建训练镜像"
	@echo "make prepare      下载指引+转YOLO格式(SeaDronesSee+AFO)"
	@echo "make train        YOLOv12+P2 训练(imgsz1024,默认骨干 yolo12-p2)"
	@echo "make eval         mAP + 分桶小目标召回"
	@echo "make export       PyTorch->ONNX(+onnxruntime自检)"
	@echo "make crossdomain  陆->海 域差评价(--demo)"
	@echo "make run / shell  进容器交互"
	@echo "(Orin端) 在 Jetson 上原生: python trt_infer_orin.py --onnx best.onnx --fp16 --benchmark"

build:
	docker build -t $(IMAGE) .

run shell:
	docker run -it --rm $(GPU) -v $(CURDIR):/workspace -v $(DATA):/workspace/datasets $(IMAGE) bash

prepare:
	$(RUNPC) python prepare_data.py --root ./datasets --step convert

train:
	$(RUNPC) python train.py --data configs/searescue.yaml --p2 --epochs 100 --imgsz 1024

eval:
	$(RUNPC) python eval.py --weights runs/detect/train/weights/best.pt --data configs/searescue.yaml

export:
	$(RUNPC) python export_onnx.py --weights runs/detect/train/weights/best.pt --imgsz 1024

crossdomain:
	$(RUNPC) python crossdomain_eval.py --demo

clean:
	-docker rmi $(IMAGE)
