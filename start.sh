#!/bin/bash
# Render start script
echo "Starting with Gunicorn..."
gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 --access-logfile - --error-logfile - server:app
