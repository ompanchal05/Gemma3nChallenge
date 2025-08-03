document.addEventListener("DOMContentLoaded", function () {
    const leafButton = document.getElementById("leaf-btn");
    const resultDiv = document.getElementById("leaf-result");
    const voiceBtn = document.getElementById("voice-btn");
    const loader = document.createElement("div");

    // 🔄 Loader Style
    loader.style.display = "none";
    loader.style.width = "50px";
    loader.style.height = "50px";
    loader.style.border = "6px solid #00ff7f";
    loader.style.borderTop = "6px solid transparent";
    loader.style.borderRadius = "50%";
    loader.style.margin = "15px auto";
    loader.style.animation = "spin 1s linear infinite";
    loader.id = "loader";

    resultDiv.parentNode.insertBefore(loader, resultDiv);

    // 🎬 Loader Animation Keyframes
    const style = document.createElement("style");
    style.innerHTML = `
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .success {
            color: #00ff7f;
            font-weight: bold;
        }
        .error {
            color: #ff4444;
            font-weight: bold;
        }
    `;
    document.head.appendChild(style);

    // 🔘 Click Event for Leaf Button
    leafButton.addEventListener("click", async () => {
        loader.style.display = "block";
        resultDiv.innerText = "Fetching info... ⏳";
        resultDiv.className = "";

        try {
            const response = await fetch("/leaf_info");
            if (!response.ok) throw new Error("Server Error");

            const data = await response.text();
            loader.style.display = "none";

            resultDiv.innerText = data;
            resultDiv.className = "success";

            // ✅ Auto-save last info
            localStorage.setItem("lastLeafInfo", data);

        } catch (error) {
            loader.style.display = "none";
            resultDiv.innerText = "❌ Error fetching data. Click Retry.";
            resultDiv.className = "error";

            // 🔁 Retry Button
            const retryBtn = document.createElement("button");
            retryBtn.innerText = "🔄 Retry";
            retryBtn.style.background = "#ffcc00";
            retryBtn.style.color = "#000";
            retryBtn.style.padding = "8px 15px";
            retryBtn.style.borderRadius = "20px";
            retryBtn.style.marginTop = "10px";
            retryBtn.style.cursor = "pointer";

            retryBtn.addEventListener("click", () => leafButton.click());
            resultDiv.appendChild(document.createElement("br"));
            resultDiv.appendChild(retryBtn);

            console.error("Error:", error);
        }
    });

    // 🔊 Play Voice Button
    if (voiceBtn) {
        voiceBtn.addEventListener("click", () => {
            const text = resultDiv.innerText;
            if (!text || text.includes("Fetching")) {
                alert("⚠️ No info available yet!");
                return;
            }

            fetch(`/play_voice?text=${encodeURIComponent(text)}`)
                .then(res => res.json())
                .then(data => {
                    const audio = new Audio(data.audio_url);
                    audio.play();
                })
                .catch(err => {
                    console.error("Voice Error:", err);
                    alert("❌ Voice playback failed");
          });
        });
    }

    // 🔁 Auto-load last saved info
    const lastInfo = localStorage.getItem("lastLeafInfo");
    if (lastInfo) {
        resultDiv.innerText = lastInfo + " (🕘 Last Saved)";
    }

    // ⏳ Auto-refresh every 60 sec
    setInterval(() => {
        console.log("🔄 Auto-refreshing leaf info...");
        leafButton.click();
    }, 60000);
});
