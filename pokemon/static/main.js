class ArtTherapyApp {
  constructor() {
    this.canvas = new fabric.Canvas("artCanvas", {
      isDrawingMode: true,
      width: 800,
      height: 600,
      backgroundColor: "#ffffff",
    });

    this.initWebcam();
    this.initBrushes();
    this.initEventListeners();
    this.initExportHandlers();
    this.detectEmotionLoop();
  }

  initWebcam() {
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      this.video = document.getElementById("webcam");
      this.video.srcObject = stream;
    });
  }

  initBrushes() {
    this.brushes = {
      pencil: new fabric.PencilBrush(this.canvas),
      brush: new fabric.CircleBrush(this.canvas),
      spray: new fabric.SprayBrush(this.canvas),
      eraser: new fabric.EraserBrush(this.canvas),
      calligraphy: new fabric.PatternBrush(this.canvas),
      highlighter: new fabric.CircleBrush(this.canvas),
    };

    this.setBrushProperties("pencil", {
      width: 3,
      color: "#2D2D2D",
      opacity: 0.9,
    });

    this.setBrushProperties("brush", {
      width: 10,
      color: "#4A4E69",
      opacity: 0.7,
    });

    this.setBrushProperties("spray", {
      width: 20,
      density: 4,
      color: "#9A8C98",
    });

    this.setBrushProperties("calligraphy", {
      width: 15,
      opacity: 0.8,
      color: "#22223B",
    });

    this.setBrushProperties("highlighter", {
      width: 20,
      color: "rgba(255,255,0,0.4)",
    });

    this.currentBrush = "brush";
    this.brushSize = 10;
  }

  setBrushProperties(brushName, properties) {
    Object.assign(this.brushes[brushName], properties);
  }

  initEventListeners() {
    document.querySelectorAll(".tool-btn").forEach((btn) => {
      btn.addEventListener("click", () => this.selectTool(btn.dataset.brush));
    });

    document.getElementById("brushSize").addEventListener("input", (e) => {
      this.brushSize = parseInt(e.target.value);
      this.updateBrushSize();
    });

    this.canvas.on("mouse:down", (opt) => this.handlePressure(opt));
    this.canvas.on("mouse:move", (opt) => this.handlePressure(opt));
  }

  handlePressure(opt) {
    if (opt.pointer && opt.pointer.pressure) {
      const pressure = opt.pointer.pressure;
      this.canvas.freeDrawingBrush.width = this.brushSize * pressure * 2;
    }
  }

  updateBrushSize() {
    this.canvas.freeDrawingBrush.width = this.brushSize;
  }

  selectTool(brushName) {
    this.currentBrush = brushName;
    this.canvas.freeDrawingBrush = this.brushes[brushName];
    this.updateBrushSize();
    this.updateActiveTool();
  }

  updateActiveTool() {
    document.querySelectorAll(".tool-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.brush === this.currentBrush);
    });
  }

  initExportHandlers() {
    document.getElementById("exportPng").addEventListener("click", () => {
      const dataURL = this.canvas.toDataURL({ format: "png" });
      this.downloadImage(dataURL, "artwork.png");
    });

    document.getElementById("exportJpg").addEventListener("click", () => {
      const dataURL = this.canvas.toDataURL({ format: "jpeg", quality: 0.9 });
      this.downloadImage(dataURL, "artwork.jpg");
    });

    document.getElementById("exportSvg").addEventListener("click", () => {
      const svg = this.canvas.toSVG();
      const blob = new Blob([svg], { type: "image/svg+xml" });
      this.downloadBlob(blob, "artwork.svg");
    });
  }

  downloadImage(dataURL, filename) {
    const link = document.createElement("a");
    link.download = filename;
    link.href = dataURL;
    link.click();
  }

  downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    this.downloadImage(url, filename);
    URL.revokeObjectURL(url);
  }

  async detectEmotionLoop() {
    this.faceModel = await blazeface.load();
    requestAnimationFrame(() => this.detectEmotion());
  }

  async detectEmotion() {
    const predictions = await this.faceModel.estimateFaces(this.video, false);
    if (predictions.length > 0) {
      const emotion = this.analyzeEmotion(predictions[0].landmarks);
      this.updateSuggestions(emotion);
    }
    requestAnimationFrame(() => this.detectEmotion());
  }

  analyzeEmotion(landmarks) {
    // Implement proper emotion analysis here
    return "calm";
  }

  updateSuggestions(emotion) {
    const suggestions = {
      calm: { colors: ["#C9ADA7", "#9A8C98"], tools: ["brush", "calligraphy"] },
      happy: {
        colors: ["#FFD700", "#FF69B4"],
        tools: ["spray", "highlighter"],
      },
      sad: { colors: ["#4A4E69", "#22223B"], tools: ["pencil", "brush"] },
    };

    const { colors, tools } = suggestions[emotion] || suggestions.calm;
    this.updateColorSuggestions(colors);
    this.updateToolSuggestions(tools);
  }

  updateColorSuggestions(colors) {
    const colorPalette = document.getElementById("colorSuggestions");
    colorPalette.innerHTML = colors
      .map(
        (color) => `
      <div class="color-swatch" style="background: ${color}" 
           data-color="${color}"></div>
    `
      )
      .join("");

    colorPalette.querySelectorAll(".color-swatch").forEach((swatch) => {
      swatch.addEventListener("click", () => {
        this.canvas.freeDrawingBrush.color = swatch.dataset.color;
      });
    });
  }

  updateToolSuggestions(tools) {
    document.querySelectorAll(".tool-btn").forEach((btn) => {
      btn.classList.toggle("suggested", tools.includes(btn.dataset.brush));
    });
  }
}

// Initialize application
window.addEventListener("DOMContentLoaded", () => {
  new ArtTherapyApp();
});
