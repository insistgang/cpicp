# C_baseline · 一键化（容器起服务 / 跑管线 / 自检）
# 可运行性 gate: make build && make run  → 浏览器/curl 访问 :8000 即"可运行"
# 离线自证: make selftest（不需 .env / 不联网，验证镜像本身可用）

IMAGE      ?= ipo-risk-c
IMAGE_SLIM ?= ipo-risk-c-slim
PORT  ?= 8000
RUN    = docker run --rm -v $(CURDIR):/app $(IMAGE)
RUNENV = docker run --rm --env-file .env -v $(CURDIR):/app $(IMAGE)

.PHONY: help build run selftest parse index qa agents eval shell clean build-slim run-slim selftest-slim
help:
	@echo "make build       构建完整服务镜像(含 MinerU/LightRAG，较慢)"
	@echo "make run         起 FastAPI 预警服务(:$(PORT))  ← 可运行性 gate"
	@echo "make build-slim  构建轻量验机镜像(无 MinerU/torch，几分钟)"
	@echo "make run-slim    轻量镜像起 API(:$(PORT)，无需 .env)  ← 评审快速验机"
	@echo "make selftest    离线自检(eval_metrics --selftest，不需.env)"
	@echo "make parse      MinerU 解析样例招股书 -> out/demo.json"
	@echo "make index      LightRAG 建索引(需 .env)"
	@echo "make qa         风险问答+证据回链(需 .env)"
	@echo "make agents     多智能体编排 --demo(离线可跑)"
	@echo "make eval       按赛题书口径评测(需 out/pred.json + samples/gold.json)"

build:
	docker build -t $(IMAGE) .

run:
	docker run -it --rm -p $(PORT):8000 --env-file .env -v $(CURDIR):/app $(IMAGE)

selftest:
	$(RUN) python eval_metrics.py --selftest

parse:
	$(RUN) python parse_prospectus.py --pdf samples/demo_prospectus.pdf --out out/demo.json

index:
	$(RUNENV) python build_index.py --doc out/demo.json --workdir out/rag_demo

qa:
	$(RUNENV) python risk_qa.py --workdir out/rag_demo --q "是否存在对赌/赎回条款？现金消耗压力如何？"

agents:
	$(RUN) python agents_graph.py --doc out/demo.json --demo

eval:
	$(RUN) python eval_metrics.py --pred out/pred.json --gold samples/gold.json

shell:
	docker run -it --rm --env-file .env -v $(CURDIR):/app $(IMAGE) bash

# ---- 轻量验机（无 MinerU/torch，几分钟构建；API 占位管线可运行，无需 .env）----
build-slim:
	docker build -f Dockerfile.slim -t $(IMAGE_SLIM) .

run-slim:
	docker run -it --rm -p $(PORT):8000 -v $(CURDIR):/app $(IMAGE_SLIM)

selftest-slim:
	docker run --rm $(IMAGE_SLIM) python eval_metrics.py --selftest

clean:
	-docker rmi $(IMAGE) $(IMAGE_SLIM)
