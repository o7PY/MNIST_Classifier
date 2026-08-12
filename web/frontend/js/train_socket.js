function connectTrainingSocket(jobId, { onMessage, onClose } = {}) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws/train/${encodeURIComponent(jobId)}`);

  ws.addEventListener("message", (ev) => {
    try {
      const data = JSON.parse(ev.data);
      if (onMessage) onMessage(data);
    } catch (err) {
      console.error("Bad training message:", err);
    }
  });

  ws.addEventListener("close", () => {
    if (onClose) onClose();
  });

  return ws;
}

window.connectTrainingSocket = connectTrainingSocket;
