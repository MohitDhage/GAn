#!/bin/bash
# Quick Start Script for Complete Backend Stack
# Run this to start all services in the correct order

echo "=================================================="
echo "3D Asset Generation API - Quick Start"
echo "=================================================="
echo ""

# Check if Redis is running
echo "Checking Redis..."
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running!"
    echo "   Start Redis with: redis-server"
    exit 1
fi
echo "✓ Redis is running"
echo ""

# Check if Python dependencies are installed
echo "Checking Python dependencies..."
if ! python -c "import fastapi" 2>/dev/null; then
    echo "❌ FastAPI not installed!"
    echo "   Install with: pip install -r requirements_api.txt --break-system-packages"
    exit 1
fi
echo "✓ FastAPI dependencies installed"
echo ""

# Display service URLs
echo "=================================================="
echo "Services will be available at:"
echo "=================================================="
echo "  FastAPI:     http://localhost:8000"
echo "  API Docs:    http://localhost:8000/docs"
echo "  Health:      http://localhost:8000/health"
echo ""
echo "=================================================="
echo "Starting services..."
echo "=================================================="
echo ""

# Instructions for manual startup
echo "Open 2 terminal windows and run:"
echo ""
echo "Terminal 1 - Celery Worker:"
echo "  celery -A celery_app worker --loglevel=info --pool=solo"
echo ""
echo "Terminal 2 - FastAPI Server:"
echo "  python main.py"
echo ""
echo "Then test with:"
echo "  python test_api.py test_image.png"
echo ""
