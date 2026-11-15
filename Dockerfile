# Stage 1: Build TypeScript/React Frontier Research Workbench
FROM node:22-slim AS ui-builder
WORKDIR /build/ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

# Stage 2: C++20 + PyTorch CPU + AxiomZero Runtime
FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY csrc/ ./csrc/
COPY src/ ./src/
COPY benchmarks/ ./benchmarks/
COPY --from=ui-builder /build/ui/dist ./ui/dist

RUN uv sync --no-dev

EXPOSE 8030
CMD ["uv", "run", "axiom", "serve", "--host", "0.0.0.0", "--port", "8030"]
