#!/bin/bash

# Create required directories
mkdir -p sessions output

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating sample .env file. Please edit with your credentials."
    cat > .env << EOF
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_PHONE_NUMBER=your_phone_number_with_country_code
OPENAI_API_KEY=your_openai_api_key
EOF
    echo ".env file created. Please edit it with your credentials."
else
    echo ".env file already exists."
fi

echo "Setup completed. You now have two options for running the bot:"
echo ""
echo "Option 1: Full Setup (with Coqui TTS for high-quality speech)"
echo "  docker-compose up"
echo ""
echo "Option 2: Simplified Setup (Google TTS only, less resource intensive)"
echo "  docker-compose -f docker-compose.simple.yml up"
echo ""
echo "=== IMPORTANT AUTHENTICATION INSTRUCTIONS ==="
echo -e "\033[1;31mDO NOT USE DETACHED MODE (-d flag) FOR THE FIRST RUN!\033[0m"
echo "When you first run the container, you MUST run in interactive mode"
echo "to authenticate with Telegram:"
echo ""
echo "  # CORRECT (interactive mode):"
echo "  docker-compose up"
echo ""
echo "  # INCORRECT (detached mode - won't work for first run):"
echo "  docker-compose up -d"
echo ""
echo "Watch the terminal for a prompt that looks like:"
echo "  Enter the code you received: "
echo ""
echo "At this point, check your Telegram app for the verification code and enter it."
echo "After successful authentication, you can use detached mode for future runs."
echo ""
echo "See docker-README.md for more details." 