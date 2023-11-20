import React, { useState, useEffect } from "react";
import "./App.css";
import { Configuration, OpenAIApi } from "openai";

const configuration = new Configuration({
  organization: "org-0nmrFWw6wSm6xIJXSbx4FpTw",
  apiKey: "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",
});
const openai = new OpenAIApi(configuration);

function App() {
  const [message, setMessage] = useState("");
  const [image, setImage] = useState(null);
  const [chats, setChats] = useState([]);
  const [isTyping, setIsTyping] = useState(false);

  useEffect(() => {
    const delayBeforeHello = 1000; // Adjust the delay in milliseconds

    setTimeout(() => {
      const systemMessage =
        "Hello! This is your Science Tutor. I can provide instant and expert answers to K12 science questions that you may have in different domains such as natural, social and language science. Feel free to ask me any questions you have and upload an image to start!";
      let delay = 0;
      const typingSpeed = 10; // Adjust the typing speed (milliseconds per character)

      // Simulate typing effect
      for (let i = 0; i < systemMessage.length; i++) {
        setTimeout(() => {
          setChats([
            {
              role: "assistant",
              content: systemMessage.slice(0, i + 1),
            },
          ]);

          // Scroll to the bottom after each character is added
          scrollTo(0, 1e10);
        }, delay);
        delay += typingSpeed * (i === systemMessage.length - 1 ? 1 : 2); // Longer delay after the last character
      }
    }, delayBeforeHello);
  }, []);

  const roleIcons = {
    user: "/student.png", // Replace with the actual path or URL for user icon
    assistant: "/teacher.png", // Replace with the actual path or URL for assistant icon
  };

  const chat = async (e) => {
    e.preventDefault();

    if (!message && !image) return;

    setIsTyping(true);
    scrollTo(0, 1e10);

    let msgs = chats;

    // Add text message to messages
    if (message) {
      msgs.push({ role: "user", content: message });
    }

    // Add image to messages
    if (image) {
      msgs.push({ role: "user", content: image });
    }

    // Example of a POST request with JSON data
    axios.post('http://f0e9-34-125-132-46.ngrok-free.app/chat', {
      prompt: message,
      image: image,
      history: "",
    })
    .then(response => {
      // Process the data received from the API after POST request
      console.log(response.data);
    })
    .catch(error => {
      console.error('Axios error:', error);
    });

    setChats(msgs);
    setMessage("");
    setImage(null);

    // Create chat completion
    await openai
      .createChatCompletion({
        model: "gpt-3.5-turbo",
        messages: [
          {
            role: "assistant",
            content: "You are a EbereGPT. You can help with graphic design tasks",
          },
          ...msgs,
        ],
      })
      .then((res) => {
        msgs.push({ role: "assistant", content: res.data.choices[0].message });
        setChats(msgs);
        setIsTyping(false);
        scrollTo(0, 1e10);
      })
      .catch((error) => {
        console.log(error);
      });
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(URL.createObjectURL(file));
    }
  };

  return (
    <div className="app-container">
    {/* //   <aside className="sidebar">
    //     <h2>Menu</h2>
    //   </aside> */}

      <main className="main-container">
        <h1>ScienceTutor</h1>

        <section>
          {chats && chats.length
            ? chats.map((chat, index) => (
                <div key={index} className={chat.role === "user" ? "user_msg" : "assistant_msg"}>
                  <div className="role-icon">
                    <img src={roleIcons[chat.role]} alt={chat.role} />
                  </div>
                  {chat.content.startsWith("blob:") ? (
                    <img src={chat.content} alt="Uploaded" className="uploaded-image" />
                  ) : (
                    <p>{chat.content}</p>
                  )}
                </div>
              ))
            : ""}
        </section>

        <form className="message-form" onSubmit={chat}>
          <input
            type="text"
            name="message"
            value={message}
            placeholder="Type a message here..."
            onChange={(e) => setMessage(e.target.value)}
          />
          <input
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
          />
          <button type="submit" className="send-button">
            <img src="/send.png" alt="Send" />
          </button>
        </form>
      </main>
    </div>
  );
}

export default App;
