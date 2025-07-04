# Formula 1 AI Copilot

Formula 1 AI Copilot is a modular, AI-driven decision support system designed for race engineers, strategists, and performance analysts. The system integrates modern machine learning, LLMs, computer vision, and real-time data engineering to support both competitive decision-making and long-term performance insights in Formula 1 and sim racing environments.

This project is structured for scalability as a research platform and extensibility as a commercial or SaaS-grade system.

## Overview

This system combines multiple subsystems, each solving a specific class of problems in motorsport operations. It integrates telemetry, team radio, race footage, car component data, and external conditions into an intelligent layer of analysis and recommendations.

Use cases range from in-race pit strategy optimization to long-term reliability forecasting and sim-to-real transfer learning.

## Key Modules

- **Strategy Optimizer**: Real-time tire and pit stop strategy using LSTM models and Bayesian optimization.
- **Ghost Car Visualizer**: 3D lap comparison tool using telemetry and Open3D/NeRF for driver benchmarking.
- **Racing Line Generator**: Suggests optimal racing lines and adjustments based on track, tire pressure, and environmental data.
- **Emotion Detection from Team Radio**: Transcription and classification of driver communication using Whisper and audio emotion models.
- **Track Limit Breach Detection**: Automated detection of rule violations using video analysis and computer vision (YOLOv8 + OpenCV).
- **Setup Optimization**: Uses reinforcement learning and Bayesian optimization to suggest car setups for various conditions.
- **Parts Reliability Forecasting**: Time-series models predict failure probabilities based on usage and track demands.
- **Driver-Car Compatibility Modeling**: Matches driver behavior to car configuration styles for better pairing.
- **Competitor Intelligence Engine**: Extracts insights from public data to model and predict rival team strategies.
- **Sponsor Visibility and Fan Sentiment Tracking**: Combines CV and NLP to assess brand exposure and fan reaction.
- **Multi-Agent Team Radio Reasoning**: LLM-driven reasoning engine to understand and summarize team communications.
- **Sim-to-Real Transfer Engine**: Facilitates knowledge transfer between sim environments and real-world performance.

## Architecture

The architecture is modular and built around streamable, queryable, and event-driven data. Core components include:

- Real-time ingestion via Kafka and Airbyte
- Processing and transformation via dbt and DuckDB
- Machine learning models in PyTorch, Scikit-learn, and XGBoost
- CV pipelines using YOLOv8, SAM, OpenCV, Open3D
- Audio/NLP pipelines using Whisper, wav2vec, and pyannote
- LLM interface via LangChain, GPT-4o, Claude 3, and Qdrant for vector retrieval
- A FastAPI backend and TypeScript/React frontend

## Project Structure

f1-ai-copilot/
├── core_modules/
│   ├── strategy_optimizer/
│   ├── ghost_car_visualizer/
│   ├── emotion_detection/
│   ├── racing_line_generator/
│   ├── rule_compliance_agent/
│   ├── setup_optimizer/
│   ├── reliability_forecaster/
│   ├── driver_compatibility/
│   ├── competitor_intel/
│   ├── sim_to_real_engine/
│   ├── logistics_optimizer/
│   ├── sponsor_tracker/
│   └── team_radio_agent/
│
├── data_pipeline/
│   ├── kafka/
│   ├── airbyte/
│   └── dbt/
│
├── dashboard/                # FastAPI backend + React frontend
├── llm_interface/            # LangChain + Qdrant-based retrieval + agent tools
├── scripts/                  # Utility scripts and orchestration
├── notebooks/                # Jupyter notebooks for experimentation
├── tests/                    # Unit/integration tests
├── config/                   # Global configuration files
│
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── .env.example
└── README.md


