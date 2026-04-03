#official OpenEnv base image
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="data-cleaning-openenv"
LABEL org.opencontainers.image.description="OpenEnv tabular data cleaning environment"
LABEL space_sdk_type="openenv"

WORKDIR /app

#Install environment-specific dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

#Copy environment code
COPY models.py          ./models.py
COPY client.py          ./client.py
COPY pyproject.toml     ./pyproject.toml
COPY server/            ./server/

#Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

#HuggingFace Spaces requires port 7860
EXPOSE 7860

#Non-root user (security)
RUN useradd -m appuser
USER appuser

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]