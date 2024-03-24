# Use multi-stage builds to keep the final image clean and minimal
# Stage 1: Build Stage
FROM python:3.9.17-slim-bullseye as builder

# Install dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt 

# Stage 2: Final Stage
FROM python:3.9.17-slim-bullseye

# Copy installed Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.9 /usr/local/lib/python3.9
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only the necessary source files and scripts
COPY src /opt/src
COPY entry_point.sh /opt/
RUN chmod +x /opt/entry_point.sh

# Create a non-root user and switch to it
RUN useradd -r -u 1000 -g users myuser
USER 1000

WORKDIR /opt/src

# Set environment variables
ENV MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    PATH="/opt/app:${PATH}"

# Set the entrypoint
ENTRYPOINT ["/opt/entry_point.sh"]
