const express = require("express");
const ollama = require("ollama");

const app = express();

const PORT = 8080;

// Parse JSON request bodies
app.use(express.json());

// POST /chat
app.post("/chat", async (req, res) => {
    try {
        const { message } = req.body;

        // Validate message
        if (!message) {
            return res.status(400).json({
                error: "Message is required"
            });
        }

        // Send message to Ollama
        const response = await ollama.chat({
            model: "gemma3:1b",
            messages: [
                {
                    role: "user",
                    content: message
                }
            ]
        });

        // Return response
        res.json({
            response: response.message.content
        });

    } catch (error) {
        console.error(error);

        res.status(500).json({
            error: "Something went wrong",
            details: error.message
        });
    }
});

// Start server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});