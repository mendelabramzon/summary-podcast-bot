# Running the Telegram Chat to Podcast Bot with Docker

This guide explains how to run the Telegram Chat to Podcast Bot using Docker, which simplifies setup and eliminates dependency conflicts.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your system
- [Docker Compose](https://docs.docker.com/compose/install/) installed on your system
- Telegram API credentials
- OpenAI API key

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/telegram-chat-podcast-bot.git
   cd telegram-chat-podcast-bot
   ```

2. Create a `.env` file in the project root with your credentials:
   ```
   TELEGRAM_API_ID=your_api_id
   TELEGRAM_API_HASH=your_api_hash
   TELEGRAM_PHONE_NUMBER=your_phone_number_with_country_code
   OPENAI_API_KEY=your_openai_api_key
   ```

3. Create required directories:
   ```bash
   mkdir -p sessions output
   ```

## Running with Docker

You have two options for running the bot with Docker:

### ⚠️ Important: First-time Authentication

**The first time you run the bot, you MUST run it in interactive mode (without the `-d` flag)** to authenticate with Telegram:

```bash
# CORRECT for first run (interactive mode)
docker-compose up

# INCORRECT for first run (detached mode)
docker-compose up -d  # Won't work for authentication!
```

When prompted, enter the verification code sent to your Telegram app.

### Option 1: Full Setup (with Coqui TTS)

This option includes all features, including high-quality text-to-speech with Coqui TTS.
It requires more resources but provides better quality speech synthesis.

1. Build and start the container:
   ```bash
   docker-compose up
   ```

   After successful authentication, you can use detached mode for future runs:
   ```bash
   docker-compose up -d
   ```

2. Follow the logs if running in background:
   ```bash
   docker-compose logs -f
   ```

3. Stop the container:
   ```bash
   docker-compose down
   ```

### Option 2: Simplified Setup (Google TTS only)

This option uses only Google TTS for speech synthesis, which requires fewer resources
and is easier to set up.

1. Build and start the simplified container:
   ```bash
   docker-compose -f docker-compose.simple.yml up
   ```

   After successful authentication, you can use detached mode for future runs:
   ```bash
   docker-compose -f docker-compose.simple.yml up -d
   ```

2. Follow the logs if running in background:
   ```bash
   docker-compose -f docker-compose.simple.yml logs -f
   ```

3. Stop the container:
   ```bash
   docker-compose -f docker-compose.simple.yml down
   ```

### Using Docker Directly

If you prefer to run Docker commands directly:

1. Build the Docker image (full version):
   ```bash
   docker build -t telegram-podcast-bot .
   ```

   Or build the simplified version:
   ```bash
   docker build -f Dockerfile.simple -t telegram-podcast-bot-simple .
   ```

2. Run the container:
   ```bash
   docker run -it --rm \
     -v "$(pwd)/.env:/app/.env" \
     -v "$(pwd)/sessions:/app/sessions" \
     -v "$(pwd)/output:/app/output" \
     telegram-podcast-bot
   ```

   Note the `-it` flags are required for interactive authentication.

## Important Notes

1. **First-time authentication**: When first running the container, you'll need to authenticate with Telegram. The session file will be saved in the `sessions` directory on your host machine.

2. **Output files**: Generated podcast audio files and text summaries will be saved to the `output` directory on your host machine.

3. **Interactive mode**: The bot requires interaction to select chats and configure options. The Docker container is configured with interactive mode enabled.

4. **Resource requirements**: The full setup with Coqui TTS requires at least 4GB of memory. The simplified setup has lower requirements.

## Troubleshooting

- **Can't authenticate**: Make sure you're running Docker without the `-d` flag for the first run. You must be able to see and interact with the terminal.
  
- **Container exits immediately**: Make sure your `.env` file has the correct credentials and is properly mounted.
  
- **Audio generation fails**: The container includes ffmpeg and other audio dependencies, but you may need to update the Dockerfile if you encounter specific issues.

- **Session issues**: If you encounter authentication problems, try removing the saved session file and restart the container to re-authenticate.

- **Out of memory errors**: If you're seeing memory-related errors with the full setup, try using the simplified version or increase the memory allocation in docker-compose.yml. 