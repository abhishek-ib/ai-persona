# AI Persona Bot 🤖

An intelligent chatbot that learns from Slack message history to respond as specific users, creating personalized AI personas using advanced vector search and Google Gemini AI.

## Features ✨

- **Data Normalization**: Processes Slack export data and cleans message content
- **Scalable Vector Search**: Uses Pinecone for cloud-based vector storage (recommended) or FAISS for local storage
- **Neural Embeddings**: all-MiniLM-L6-v2 embeddings for high-quality similarity search
- **Persona Generation**: Creates responses that match individual user communication styles
- **Google Gemini Integration**: Leverages advanced AI for natural language generation
- **Interactive Chat**: Real-time conversation interface
- **Response Caching**: Improves performance with intelligent caching
- **Multi-user Support**: Switch between different user personas seamlessly
- **Multiple Storage Options**: Pinecone (scalable), FAISS (local), or TF-IDF (minimal)

## Quick Start 🚀

### 1. Installation

```bash
# Clone or download the project
cd ai-persona

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup API Keys

**Pinecone API Key (Recommended for scalability):**
Get your API key from [Pinecone](https://app.pinecone.io/) and set it as an environment variable:

```bash
export PINECONE_API_KEY="your-pinecone-api-key-here"
```

**Google AI API Key:**
Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey) and set it as an environment variable:

```bash
export GOOGLE_AI_API_KEY="your-google-ai-api-key-here"
```

Or create a `.env` file:
```
PINECONE_API_KEY=your-pinecone-api-key-here
GOOGLE_AI_API_KEY=your-google-ai-api-key-here
```

### 3. Prepare Your Data

Place your Slack export data in the `data/` directory:
```
data/
├── users.json          # User information
├── channels.json       # Channel information  
├── D097DGGD37B/       # Channel/DM directory
│   ├── 2025-07-22.json
│   ├── 2025-07-24.json
│   └── ...
└── attachments/       # File attachments
```

### 4. Run the Bot

**Interactive Mode:**
```bash
python ai_persona_bot.py --interactive
```

**Single Query:**
```bash
python ai_persona_bot.py --user "Jaden Lee" --query "How's the project going?"
```

**Rebuild Index (if data changed):**
```bash
python ai_persona_bot.py --rebuild-index --interactive
```

## Usage Guide 📖

### Interactive Commands

Once in interactive mode, you can use these commands:

- `help` - Show available commands
- `users` - List available personas
- `persona <name>` - Select a user to respond as
- `search <query>` - Find similar messages
- `quit` - Exit the bot

### Example Session

```
🤖 AI Persona Bot - Interactive Mode

[No persona selected] > users
👥 Available Users/Personas:
  - Jaden Lee (45 messages)
  - Paul Hoang (32 messages)

[No persona selected] > persona Jaden Lee
✅ Now responding as: Jaden Lee

[Jaden Lee] > How do you feel about the new feature?
💬 I think it's pretty solid! The implementation looks clean and should handle the edge cases we discussed. Looking forward to seeing how users respond to it.

[Jaden Lee] > search deployment issues
🔍 Found 3 similar messages:
1. [2025-09-03 14:30] Paul Hoang: Had some deployment issues with the CSV reader...
   Similarity: 0.847
```

## Architecture 🏗️

### Components

1. **Data Normalizer** (`data_normalizer.py`)
   - Processes Slack JSON exports
   - Cleans and structures message data
   - Handles user information and metadata

2. **Vector Store** (`vector_store.py`)
   - Creates embeddings using all-MiniLM-L6-v2
   - Builds FAISS index for similarity search
   - Manages message metadata and context

3. **Gemini Client** (`gemini_client.py`)
   - Integrates with Google Gemini API
   - Generates persona-based responses
   - Handles prompt engineering and context

4. **Main Bot** (`ai_persona_bot.py`)
   - Orchestrates all components
   - Provides interactive interface
   - Manages user sessions and caching

### Data Flow

```
Slack Data → Normalizer → Vector Store → Similarity Search → Gemini → Response
     ↓            ↓            ↓              ↓             ↓         ↓
  JSON files → Clean text → Embeddings → Context → Prompt → AI Response
```

## Configuration ⚙️

### Environment Variables

- `GOOGLE_AI_API_KEY` - Your Google AI API key (required)

### Model Settings

You can customize the models used:

```python
# In vector_store.py
vector_store = VectorStore(model_name="all-MiniLM-L6-v2")

# In gemini_client.py  
client = GeminiClient(model_name="gemini-1.5-flash")
```

### Generation Parameters

Adjust response generation in `gemini_client.py`:

```python
generation_config = genai.types.GenerationConfig(
    temperature=0.7,      # Creativity (0.0-1.0)
    top_p=0.8,           # Nucleus sampling
    top_k=40,            # Top-k sampling
    max_output_tokens=512 # Response length
)
```

## API Reference 📚

### AIPersonaBot Class

```python
bot = AIPersonaBot(data_dir="data", rebuild_index=False)

# Initialize Gemini
bot.initialize_gemini(api_key="your-key")

# Chat as user
response = bot.chat_as_user("Jaden Lee", "Hello!")

# Search messages
results = bot.search_similar_messages("deployment issues")

# Get user context
context = bot.get_user_context("Paul Hoang", limit=10)
```

### Response Format

```python
{
    'success': True,
    'response': "Generated response text",
    'user_name': "Jaden Lee",
    'user_id': "U0968P53AQ4", 
    'query': "Original query",
    'timestamp': "2025-09-16T10:30:00",
    'from_cache': False
}
```

## Advanced Usage 🔧

### Custom Data Sources

To use data from other sources, implement the message format:

```python
{
    'user_id': 'unique_user_id',
    'user_name': 'Display Name',
    'content': 'Message content',
    'timestamp': 1694865000.0,
    'datetime': '2023-09-16T10:30:00'
}
```

### Batch Processing

Process messages programmatically:

```python
from ai_persona_bot import AIPersonaBot

bot = AIPersonaBot()
queries = ["How's the project?", "Any updates?", "Need help?"]

for query in queries:
    response = bot.chat_as_user("Jaden Lee", query)
    print(f"Q: {query}")
    print(f"A: {response['response']}\n")
```

### Performance Optimization

- Use `--rebuild-index` only when data changes
- Enable caching for repeated queries
- Adjust `batch_size` in vector_store.py for your hardware
- Consider using `faiss-gpu` for large datasets

## Troubleshooting 🔧

### Common Issues

**"Index files not found"**
- Run with `--rebuild-index` flag
- Ensure data directory contains valid JSON files

**"Gemini API connection failed"**
- Check your API key is set correctly
- Verify internet connection
- Ensure API key has proper permissions

**"User not found"**
- Use `users` command to see available personas
- Check user has sufficient message history (>5 messages)

**Empty or poor responses**
- Verify user has enough context messages
- Check message quality in source data
- Adjust temperature in generation config

### Performance Issues

- Large datasets: Use `faiss-gpu` instead of `faiss-cpu`
- Memory issues: Reduce batch size in vector_store.py
- Slow responses: Enable response caching
- Index building: Use SSD storage for better I/O performance

## Development 👨‍💻

### Project Structure

```
ai-persona/
├── ai_persona_bot.py      # Main bot interface
├── data_normalizer.py     # Data processing
├── vector_store.py        # Embedding & search
├── gemini_client.py       # AI integration
├── requirements.txt       # Dependencies
├── README.md             # Documentation
├── data/                 # Source data
├── normalized_messages.json  # Processed data
├── message_index.faiss   # Vector index
├── message_metadata.pkl  # Message metadata
└── response_cache.json   # Response cache
```

### Testing

```bash
# Test individual components
python data_normalizer.py
python vector_store.py  
python gemini_client.py

# Test full pipeline
python ai_persona_bot.py --user "Test User" --query "Hello"
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License 📄

This project is open source and available under the [MIT License](LICENSE).

## Support 💬

For questions, issues, or contributions:

1. Check existing GitHub issues
2. Create a new issue with detailed information
3. Include error messages and system information
4. Provide steps to reproduce problems

---

**Happy chatting with your AI personas!** 🎉
