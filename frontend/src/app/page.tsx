"use client";

import { useState, useEffect, useRef } from "react";

interface AssetsResponse {
  originals: {
    images: string[];
    videos: string[];
  };
  results: {
    images: string[];
    videos: string[];
  };
}

interface ClickPoint {
  x: number;
  y: number;
  is_positive: boolean;
}

const getSamplePromptsForAsset = (path: string | null): string[] => {
  if (!path) return [];
  if (path.includes("test_image")) return [
    // Easy — obvious objects
    "child", "shoe", "sneakers",
    // Clothing items
    "red training bib", "blue training bib", "white shirt",
    "sweatpants", "leggings",
    // Positional — left to right
    "leftmost child", "second child from left", "third child from left",
    "fourth child from left", "fifth child from left", "rightmost child",
    // By attribute — clothing color
    "child wearing red bib", "child wearing blue bib",
    "child wearing light blue sweatpants",
    "child wearing black leggings",
    // By attribute — detailed descriptions (left to right)
    "boy in striped shirt with red bib",
    "girl with long dark hair wearing red bib",
    "girl with curly hair wearing blue bib and pink shirt",
    "girl with blue headband wearing blue bib",
    "boy in white long-sleeve shirt with red bib and black sweatpants",
    "boy in blue bib with white sneakers and arms outstretched",
    // Specific clothing & accessories
    "blue headband", "pink shirt", "striped shirt",
    "black sneakers with white soles", "white sneakers",
    "dark sneakers with blue accents",
    // Pose-based
    "child running", "child standing", "child leaning forward",
  ];
  if (path.includes("truck")) return [
    // Easy — the vehicle
    "truck", "wheel", "tire", "window",
    // Vehicle body & structure
    "white pickup truck", "rear canopy", "truck bed",
    "cab window", "tinted window",
    // Wheels & trim
    "black rubber tire", "silver steel rim", "center hubcap",
    "front wheel arch", "rear wheel arch",
    // Vehicle details
    "door handle", "side-view mirror", "bull bar",
    "front bumper", "rear bumper", "tow hitch",
    "taillight", "side rail",
    // Detailed descriptions
    "white fiberglass canopy with tinted windows",
    "silver chrome tubular grille guard",
    "red and clear taillight assembly",
    "black rear bumper step",
    "black horizontal tie-down bar beneath canopy",
    // Environment — ground & structures
    "asphalt road", "shadow beneath truck",
    "concrete curb", "sidewalk",
    // Environment — background
    "red wall", "diagonal staircase line on wall",
    "balcony with railing", "white sculpted balusters",
  ];
  if (path.includes("groceries")) return [
    // Easy — large, obvious objects
    "bag", "food", "cargo area", "rear bumper",
    // Medium — specific items
    "bread", "baguette", "headrests", "red taillights", "exhaust pipe",
    // Positional — requires spatial reasoning
    "leftmost bag", "middle bag", "rightmost bag", "second bag from left",
    // Detailed descriptions — vehicle interior
    "black cargo floor mat", "beige cargo cover", "beige leather rear seats",
    "interior side panels", "ceiling dome light", "parking sensors",
    // Complex — background & environment
    "paved ground", "drainage cover", "metal support pillars",
    "trees and green foliage",
  ];
  if (path.includes("bedroom")) return [
    // Easy — large, obvious objects
    "bed", "kid", "pillow",
    // Medium — specific furniture & bedding
    "bed mattress", "striped bedspread",
    // Body parts — large to small
    "face", "head", "hair", "arms", "legs",
    "eyes", "nose", "mouth", "ears", "neck", "shoulders",
    "hands", "knees", "feet", "fingers", "toes",
    // Specific people descriptions
    "boy wearing black and white baseball shirt", "girl wearing blue dress",
    // Specific pillows
    "grey pillow", "white patterned pillow",
    "light blue textured pillow", "peach pillow",
    // Wall & decor
    "white floating shelf", "string lights", "light switch",
    "pink and white striped wallpaper",
    "book on shelf", "glass jar on shelf", "candles on shelf",
    // Complex — mirror & reflected objects
    "white-framed floor mirror", "round wall mirror with black frame",
    "black lamp", "white dresser with drawers",
    "small wooden stand", "glass bottle on dresser",
  ];
  if (path.includes("0001")) return ["person"];
  if (path.includes("dog")) return ["dog"];
  if (path.includes("player")) return ["player"];
  return [];
};

export default function Home() {
  const [assets, setAssets] = useState<AssetsResponse | null>(null);
  const [selectedAsset, setSelectedAsset] = useState<string>("images/truck.jpg");
  const [assetType, setAssetType] = useState<"image" | "video">("image");
  const [prompt, setPrompt] = useState<string>("truck");
  const [promptMode, setPromptMode] = useState<"text" | "click">("text");
  const [clicks, setClicks] = useState<ClickPoint[]>([]);
  const [clickLabel, setClickLabel] = useState<boolean>(true); // true = positive, false = negative
  
  const [loading, setLoading] = useState<boolean>(false);
  const [outputUrl, setOutputUrl] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"playground" | "browser">("playground");
  const [healthStatus, setHealthStatus] = useState<{ status: string; gpu_available: boolean } | null>(null);
  
  // Tour State
  const [tourStep, setTourStep] = useState<number | null>(null);
  const [isSidebarExpanded, setIsSidebarExpanded] = useState<boolean>(true);
  const isSidebarOpen = isSidebarExpanded || tourStep === 1;

  const imageRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    // Fetch assets and health status
    fetchAssets();
    fetchHealth();

    // Auto-start tour if never completed
    const completed = localStorage.getItem("sam3_tour_completed");
    if (!completed) {
      setTourStep(0);
    }
  }, []);

  const endTour = () => {
    localStorage.setItem("sam3_tour_completed", "true");
    setTourStep(null);
  };

  const fetchAssets = async () => {
    try {
      const res = await fetch("/api/assets");
      const data = await res.json();
      setAssets(data);
    } catch (e) {
      console.error("Failed to fetch assets", e);
    }
  };

  const fetchHealth = async () => {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      setHealthStatus(data);
    } catch (e) {
      console.error("Failed to fetch health status", e);
    }
  };

  const handleAssetChange = (path: string) => {
    setSelectedAsset(path);
    const isVideo = path.startsWith("videos") || path.toLowerCase().endsWith(".gif") || path.includes("0001");
    const type = isVideo ? "video" : "image";
    setAssetType(type);
    setOutputUrl(null);
    setClicks([]);
    setSessionId(null);
    
    // Choose suitable default prompts
    if (path.includes("truck")) setPrompt("truck");
    else if (path.includes("groceries")) setPrompt("bag");
    else if (path.includes("bedroom")) setPrompt("bed");
    else if (path.includes("0001")) setPrompt("person");
    else if (path.includes("dog")) setPrompt("dog");
    else if (path.includes("player")) setPrompt("player");
    else setPrompt("object");
  };

  const handleImageClick = (e: React.MouseEvent<HTMLImageElement>) => {
    if (promptMode !== "click" || !imageRef.current) return;
    
    const rect = imageRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    
    setClicks([...clicks, { x, y, is_positive: clickLabel }]);
  };

  const clearClicks = () => {
    setClicks([]);
    setOutputUrl(null);
    setSessionId(null);
  };

  const runTextInference = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/inference", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          asset_type: assetType,
          asset_path: selectedAsset,
          prompt: prompt,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setOutputUrl(data.output_url);
        setSessionId(data.session_id);
      } else {
        alert("Inference failed: " + data.detail);
      }
    } catch (e) {
      console.error(e);
      alert("Error running inference");
    } finally {
      setLoading(false);
    }
  };

  const runClickInference = async () => {
    if (clicks.length === 0) {
      alert("Please click on the image to add positive/negative prompts first.");
      return;
    }
    
    setLoading(true);
    try {
      const res = await fetch("/api/interactive-click", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          asset_path: selectedAsset,
          clicks: clicks,
          prompt: prompt || undefined,
          session_id: sessionId || undefined,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setOutputUrl(data.output_url);
        setSessionId(data.session_id);
      } else {
        alert("Interactive click inference failed: " + data.detail);
      }
    } catch (e) {
      console.error(e);
      alert("Error running click inference");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#07070F] text-[#F3F4F6] font-sans antialiased selection:bg-indigo-500 selection:text-white">
      {/* Background radial glow */}
      <div className="absolute top-0 left-0 w-full h-[500px] bg-gradient-to-b from-indigo-950/20 via-transparent to-transparent pointer-events-none" />
      <div className="absolute top-[20%] right-[10%] w-[300px] h-[300px] bg-purple-900/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute top-[40%] left-[5%] w-[400px] h-[400px] bg-indigo-900/10 rounded-full blur-[150px] pointer-events-none" />

      {/* Top Header */}
      <header className="relative border-b border-white/5 bg-[#0C0C16]/80 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center font-bold text-white shadow-lg shadow-indigo-500/20">
              S3
            </div>
            <div>
              <h1 className="text-lg font-semibold bg-gradient-to-r from-white via-slate-200 to-indigo-400 bg-clip-text text-transparent">
                SAM3 Studio
              </h1>
              <p className="text-[10px] text-slate-500 uppercase tracking-widest font-mono">Segment Anything v3</p>
            </div>
          </div>
          
          {/* Navigation Tabs */}
          <div className="flex space-x-1 bg-white/5 p-1 rounded-xl border border-white/5">
            <button
              onClick={() => setActiveTab("playground")}
              className={`px-4 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === "playground"
                  ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/10"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Interactive Playground
            </button>
            <button
              onClick={() => setActiveTab("browser")}
              className={`px-4 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === "browser"
                  ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/10"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Asset Gallery
            </button>
          </div>

          {/* Help Tour & System Health Indicators */}
          <div className="flex items-center space-x-3 text-xs">
            <button
              onClick={() => setTourStep(0)}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 border border-indigo-500/20 text-white transition-all shadow-md shadow-indigo-600/10 cursor-pointer"
            >
              <span>❓ Help Tour</span>
            </button>
            <div className="flex items-center space-x-2 bg-white/5 border border-white/5 px-3 py-1.5 rounded-xl">
              <span className={`w-2 h-2 rounded-full ${healthStatus ? "bg-emerald-500" : "bg-rose-500"} animate-pulse`} />
              <span className="text-slate-400">System API</span>
            </div>
            <div className="flex items-center space-x-2 bg-white/5 border border-white/5 px-3 py-1.5 rounded-xl">
              <span className={`w-2 h-2 rounded-full ${healthStatus?.gpu_available ? "bg-emerald-500" : "bg-amber-500"}`} />
              <span className="text-slate-400">GPU Acceleration</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="relative max-w-7xl mx-auto px-6 py-8">
        
        {/* Interactive Playground Tab */}
        {activeTab === "playground" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            
            {/* Left sidebar - configurations */}
            <div className={`transition-all duration-300 ${isSidebarOpen ? "lg:col-span-3" : "lg:col-span-1 lg:max-w-[80px]"} space-y-6`}>
              
              {/* Card - Select Input Asset */}
              <div className={`bg-[#0C0C16] border border-white/5 rounded-2xl ${
                isSidebarOpen ? "p-4" : "p-3"
              } shadow-xl backdrop-blur-md transition-all duration-300 ${
                tourStep === 1 ? "ring-4 ring-indigo-500 ring-offset-2 ring-offset-[#07070F] z-50 relative bg-[#121225]" : ""
              }`}>
                {isSidebarOpen ? (
                  <>
                    <div className="flex items-center justify-between mb-4 pb-2 border-b border-white/5">
                      <h2 className="text-xs font-semibold text-slate-200 flex items-center space-x-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
                        <span>Select Test Asset</span>
                      </h2>
                      {tourStep !== 1 && (
                        <button
                          onClick={() => setIsSidebarExpanded(false)}
                          className="p-1 rounded-lg bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white transition-all cursor-pointer border border-white/5 flex items-center justify-center"
                          title="Collapse Sidebar"
                        >
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                          </svg>
                        </button>
                      )}
                    </div>
                    {assets ? (
                      <div className="space-y-5">
                        <div>
                          <label className="block text-[10px] uppercase tracking-wider text-slate-500 font-mono mb-2">Images</label>
                          <div className="grid grid-cols-1 gap-2 max-h-[250px] overflow-y-auto pr-1">
                            {assets.originals.images.map((img) => (
                              <button
                                key={img}
                                onClick={() => handleAssetChange(img)}
                                title={img.split("/").pop()}
                                className={`group relative aspect-video rounded-xl overflow-hidden border bg-black/20 transition-all cursor-pointer ${
                                  selectedAsset === img
                                    ? "border-indigo-500 ring-2 ring-indigo-500/20 shadow-lg shadow-indigo-500/10"
                                    : "border-white/5 hover:border-white/10"
                                }`}
                              >
                                <img
                                  src={`/api/original/${img}`}
                                  alt={img}
                                  className="w-full h-full object-cover group-hover:scale-110 transition-all duration-300"
                                  loading="lazy"
                                />
                              </button>
                            ))}
                          </div>
                        </div>
                        
                        <div>
                          <label className="block text-[10px] uppercase tracking-wider text-slate-500 font-mono mb-2">Videos & GIFs</label>
                          <div className="grid grid-cols-1 gap-2 max-h-[250px] overflow-y-auto pr-1">
                            {assets.originals.videos.map((vid) => {
                              const isGif = vid.toLowerCase().endsWith(".gif");
                              const isFrameFolder = vid.includes("0001");
                              const isMp4 = vid.toLowerCase().endsWith(".mp4");
                              
                              return (
                                <button
                                  key={vid}
                                  onClick={() => handleAssetChange(vid)}
                                  title={vid.split("/").pop()}
                                  className={`group relative aspect-video rounded-xl overflow-hidden border bg-black/20 transition-all cursor-pointer ${
                                    selectedAsset === vid
                                      ? "border-indigo-500 ring-2 ring-indigo-500/20 shadow-lg shadow-indigo-500/10"
                                      : "border-white/5 hover:border-white/10"
                                  }`}
                                >
                                  {isGif ? (
                                    <img
                                      src={`/api/original/${vid}`}
                                      alt={vid}
                                      className="w-full h-full object-cover group-hover:scale-110 transition-all duration-300"
                                      loading="lazy"
                                    />
                                  ) : isFrameFolder ? (
                                    <img
                                      src={`/api/original/${vid}/00000.jpg`}
                                      alt={vid}
                                      className="w-full h-full object-cover group-hover:scale-110 transition-all duration-300"
                                      loading="lazy"
                                    />
                                  ) : isMp4 ? (
                                    <video
                                      src={`/api/original/${vid}`}
                                      muted
                                      loop
                                      playsInline
                                      className="w-full h-full object-cover group-hover:scale-110 transition-all duration-300 pointer-events-none"
                                    />
                                  ) : (
                                    <div className="w-full h-full flex items-center justify-center font-mono text-[9px] text-slate-500">
                                      🎥
                                    </div>
                                  )}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="py-8 flex justify-center">
                        <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                      </div>
                    )}
                  </>
                ) : (
                  <div className="flex flex-col items-center space-y-4">
                    <button
                      onClick={() => setIsSidebarExpanded(true)}
                      className="p-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition-all cursor-pointer border border-indigo-500/20 shadow-md shadow-indigo-600/10 flex items-center justify-center"
                      title="Expand Sidebar"
                    >
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                    </button>
                    <div className="text-[10px] font-mono text-slate-500 uppercase tracking-widest [writing-mode:vertical-lr] rotate-180 select-none py-2 font-semibold">
                      Assets
                    </div>
                    {selectedAsset && (
                      <div className="relative w-10 h-10 rounded-lg overflow-hidden border border-white/10 bg-black/40 shadow-md">
                        {assetType === "image" ? (
                          <img
                            src={`/api/original/${selectedAsset}`}
                            alt="Selected"
                            className="w-full h-full object-cover"
                          />
                        ) : selectedAsset.toLowerCase().endsWith(".gif") ? (
                          <img
                            src={`/api/original/${selectedAsset}`}
                            alt="Selected"
                            className="w-full h-full object-cover"
                          />
                        ) : selectedAsset.includes("0001") ? (
                          <img
                            src={`/api/original/${selectedAsset}/00000.jpg`}
                            alt="Selected"
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <video
                            src={`/api/original/${selectedAsset}`}
                            muted
                            className="w-full h-full object-cover pointer-events-none"
                          />
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>

            </div>

            {/* Right container - Visual playground */}
            <div className={`transition-all duration-300 ${isSidebarOpen ? "lg:col-span-9" : "lg:col-span-11"} space-y-6`}>
              
              <div className="bg-[#0C0C16] border border-white/5 rounded-3xl p-6 shadow-2xl backdrop-blur-lg flex flex-col min-h-[500px]">
                <div className="flex items-center justify-between border-b border-white/5 pb-4 mb-4">
                  <div className="flex items-center space-x-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-indigo-500" />
                    <h2 className="text-sm font-semibold text-white">Visual Workspace</h2>
                  </div>
                  <span className="text-xs text-slate-400 bg-white/5 px-3 py-1 rounded-xl border border-white/5">
                    {selectedAsset.split("/").pop()}
                  </span>
                </div>

                {/* Workspace Split (Input on left, Output on right) */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 flex-1 items-start justify-center">
                  
                  {/* Left Column: Original Asset */}
                  <div className={`flex flex-col items-center justify-center space-y-4 w-full p-2 rounded-2xl transition-all duration-300 ${
                    tourStep === 2 ? "ring-4 ring-indigo-500 ring-offset-2 ring-offset-[#07070F] z-50 relative bg-[#0C0C16]/90" : ""
                  }`}>
                    <span className="text-xs text-slate-400 uppercase tracking-widest font-mono">Original Asset</span>
                    <div className="relative border border-white/5 bg-black/40 rounded-2xl overflow-hidden shadow-inner group">
                      {assetType === "image" ? (
                        <div className="relative inline-block cursor-crosshair">
                          <img
                            ref={imageRef}
                            src={`/api/original/${selectedAsset}`}
                            alt="Original Asset"
                            onClick={handleImageClick}
                            className="max-h-[350px] w-auto object-contain select-none"
                          />
                          {/* Dot Overlays for click prompts */}
                          {promptMode === "click" && clicks.map((click, idx) => (
                            <div
                              key={idx}
                              style={{
                                left: `${click.x * 100}%`,
                                top: `${click.y * 100}%`
                              }}
                              className={`absolute w-3.5 h-3.5 rounded-full border-2 border-white flex items-center justify-center font-bold text-[8px] text-white shadow-md transform -translate-x-1/2 -translate-y-1/2 select-none ${
                                click.is_positive ? "bg-emerald-500" : "bg-rose-500"
                              }`}
                            >
                              {idx + 1}
                            </div>
                          ))}
                        </div>
                      ) : selectedAsset.toLowerCase().endsWith(".gif") ? (
                        <div className="relative flex items-center justify-center bg-black h-[250px] w-[350px] rounded-xl">
                          <div className="absolute text-slate-600 font-mono text-[10px] uppercase">Original GIF Reference</div>
                          <img
                            src={`/api/original/${selectedAsset}`}
                            alt="Original GIF"
                            className="h-[250px] w-auto max-w-full rounded-xl relative z-10 object-contain"
                          />
                        </div>
                      ) : (
                        <div className="relative flex items-center justify-center bg-black h-[250px] w-[350px] rounded-xl">
                          <div className="absolute text-slate-600 font-mono text-[10px] uppercase">Original Video Reference</div>
                          <video
                            src={`/api/original/${selectedAsset}`}
                            controls
                            className="h-[250px] w-auto max-w-full rounded-xl relative z-10"
                          />
                        </div>
                      )}
                    </div>
                    
                    {/* Prompt Controls Bar directly below the image */}
                    <div className="w-full bg-black/40 border border-white/5 rounded-2xl p-4 shadow-xl">
                      {/* Mode selection if image */}
                      {assetType === "image" && (
                        <div className="grid grid-cols-2 gap-2 mb-3 bg-white/5 p-1 rounded-xl border border-white/5">
                          <button
                            onClick={() => setPromptMode("text")}
                            className={`py-1 rounded-lg text-xs font-medium transition-all ${
                              promptMode === "text"
                                ? "bg-indigo-600 text-white shadow"
                                : "text-slate-400 hover:text-white"
                            }`}
                          >
                            Text Prompt
                          </button>
                          <button
                            onClick={() => setPromptMode("click")}
                            className={`py-1 rounded-lg text-xs font-medium transition-all ${
                              promptMode === "click"
                                ? "bg-indigo-600 text-white shadow"
                                : "text-slate-400 hover:text-white"
                            }`}
                          >
                            Click Prompt
                          </button>
                        </div>
                      )}

                      {promptMode === "text" || assetType === "video" ? (
                        <div className="space-y-3 w-full">
                          <div className="flex space-x-2 w-full">
                            <input
                              type="text"
                              value={prompt}
                              onChange={(e) => setPrompt(e.target.value)}
                              className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2 text-xs text-white focus:outline-none focus:border-indigo-500/50"
                              placeholder="e.g. truck, person, bed"
                            />
                            <button
                              onClick={runTextInference}
                              disabled={loading}
                              className="bg-gradient-to-r from-indigo-500 to-purple-600 text-white px-4 py-2 rounded-xl text-xs font-semibold shadow-lg hover:opacity-90 active:scale-[0.99] transition-all flex items-center justify-center space-x-2 flex-shrink-0"
                            >
                              {loading ? (
                                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                              ) : (
                                <span>Run Inference</span>
                              )}
                            </button>
                          </div>
                          {/* Suggested Sample Prompts */}
                          {getSamplePromptsForAsset(selectedAsset).length > 0 && (
                            <div className="flex flex-wrap items-center gap-1.5 pt-1">
                              <span className="text-[10px] text-slate-500 uppercase tracking-wider font-mono mr-1">Suggestions:</span>
                              {getSamplePromptsForAsset(selectedAsset).map((suggestion) => (
                                <button
                                  key={suggestion}
                                  onClick={() => setPrompt(suggestion)}
                                  className={`px-2 py-0.5 rounded-lg text-[10px] transition-all border ${
                                    prompt === suggestion
                                      ? "bg-indigo-600/20 border-indigo-500/30 text-indigo-300 font-medium"
                                      : "bg-white/5 border-white/5 text-slate-400 hover:text-white hover:bg-white/10"
                                  }`}
                                >
                                  {suggestion}
                                </button>
                              ))}
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="space-y-3">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-2">
                              <span className="text-[10px] uppercase tracking-wider text-slate-500 font-mono">Label:</span>
                              <div className="flex space-x-1 bg-white/5 p-0.5 rounded-lg border border-white/5">
                                <button
                                  onClick={() => setClickLabel(true)}
                                  className={`px-2 py-0.5 rounded-md text-[9px] font-medium transition-all ${
                                    clickLabel ? "bg-emerald-600 text-white" : "text-slate-400"
                                  }`}
                                >
                                  🟢 Positive
                                </button>
                                <button
                                  onClick={() => setClickLabel(false)}
                                  className={`px-2 py-0.5 rounded-md text-[9px] font-medium transition-all ${
                                    !clickLabel ? "bg-rose-600 text-white" : "text-slate-400"
                                  }`}
                                >
                                  🔴 Negative
                                </button>
                              </div>
                            </div>
                            
                            {clicks.length > 0 && (
                              <button onClick={clearClicks} className="text-rose-400 hover:text-rose-300 text-[10px] underline font-mono">
                                Clear ({clicks.length})
                              </button>
                            )}
                          </div>

                          <button
                            onClick={runClickInference}
                            disabled={loading || clicks.length === 0}
                            className="w-full bg-gradient-to-r from-emerald-500 to-teal-600 text-white py-2 rounded-xl text-xs font-semibold shadow-lg hover:opacity-90 active:scale-[0.99] transition-all disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                          >
                            {loading ? (
                              <>
                                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                                <span>Refining...</span>
                              </>
                            ) : (
                              <span>Inference Refinement</span>
                            )}
                          </button>
                        </div>
                      )}
                    </div>
                  </div>


                  {/* Right Column: Overlay Output */}
                  <div className={`flex flex-col items-center justify-center space-y-3 p-2 rounded-2xl transition-all duration-300 ${
                    tourStep === 3 ? "ring-4 ring-indigo-500 ring-offset-2 ring-offset-[#07070F] z-50 relative bg-[#0C0C16]/90" : ""
                  }`}>
                    <span className="text-xs text-slate-400 uppercase tracking-widest font-mono">Segmented Visualization</span>
                    <div className="border border-white/5 bg-black/40 rounded-2xl overflow-hidden shadow-inner flex items-center justify-center min-h-[250px] min-w-[250px] w-full max-w-[380px] relative">
                      {loading ? (
                        <div className="flex flex-col items-center space-y-4">
                          <div className="w-9 h-9 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                          <div className="text-slate-500 text-xs animate-pulse">Running SAM3 propagation...</div>
                        </div>
                      ) : outputUrl ? (
                        assetType === "image" ? (
                          <img
                            src={outputUrl}
                            alt="Visual Segment"
                            className="max-h-[350px] w-auto object-contain"
                          />
                        ) : (
                          <video
                            key={outputUrl}
                            src={outputUrl}
                            controls
                            autoPlay
                            loop
                            className="h-[250px] w-auto max-w-full rounded-xl"
                          />
                        )
                      ) : (
                        <div className="text-slate-600 text-xs font-mono py-12 px-6 text-center">
                          Waiting for segmentation execution...
                        </div>
                      )}
                    </div>
                  </div>

                </div>
              </div>

            </div>
          </div>
        )}

        {/* Gallery Browser Tab */}
        {activeTab === "browser" && (
          <div className="space-y-8">
            {/* Image Gallery */}
            <div className="bg-[#0C0C16] border border-white/5 rounded-3xl p-6 shadow-xl backdrop-blur-md">
              <h2 className="text-sm font-semibold text-slate-200 mb-6 flex items-center space-x-2">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
                <span>Result Image Gallery</span>
              </h2>

              {assets?.results.images && assets.results.images.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {assets.results.images.map((img) => (
                    <div key={img} className="bg-black/20 border border-white/5 rounded-2xl overflow-hidden group hover:border-indigo-500/20 transition-all">
                      <div className="aspect-[4/3] bg-black/40 flex items-center justify-center overflow-hidden relative">
                        <img
                          src={`/api/results/${img}`}
                          alt={img}
                          className="w-full h-full object-cover group-hover:scale-[1.03] transition-all duration-300"
                        />
                      </div>
                      <div className="p-4 border-t border-white/5 flex items-center justify-between">
                        <span className="text-xs text-slate-400 font-mono truncate">{img.split("/").pop()}</span>
                        <a
                          href={`/api/results/${img}`}
                          target="_blank"
                          rel="noreferrer"
                          className="text-[10px] text-indigo-400 hover:text-white uppercase tracking-wider font-mono"
                        >
                          Open Full
                        </a>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="py-12 text-center text-slate-500 text-xs font-mono">
                  No static overlay images generated yet. Run tests to populate this list.
                </div>
              )}
            </div>

            {/* Video Gallery */}
            <div className="bg-[#0C0C16] border border-white/5 rounded-3xl p-6 shadow-xl backdrop-blur-md">
              <h2 className="text-sm font-semibold text-slate-200 mb-6 flex items-center space-x-2">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-500" />
                <span>Result Video Gallery</span>
              </h2>

              {assets?.results.videos && assets.results.videos.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {assets.results.videos.map((vid) => (
                    <div key={vid} className="bg-black/20 border border-white/5 rounded-2xl overflow-hidden hover:border-indigo-500/20 transition-all">
                      <div className="aspect-video bg-black flex items-center justify-center overflow-hidden">
                        <video
                          src={`/api/results/${vid}`}
                          controls
                          className="w-full h-full object-contain"
                        />
                      </div>
                      <div className="p-4 border-t border-white/5 flex items-center justify-between">
                        <span className="text-xs text-slate-400 font-mono">{vid.split("/").pop()}</span>
                        <a
                          href={`/api/results/${vid}`}
                          target="_blank"
                          rel="noreferrer"
                          className="text-[10px] text-indigo-400 hover:text-white uppercase tracking-wider font-mono"
                        >
                          Download
                        </a>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="py-12 text-center text-slate-500 text-xs font-mono">
                  No static overlay videos generated yet. Run video tests to populate this list.
                </div>
              )}
            </div>
          </div>
        )}

      </main>

      {/* Onboarding Tour Overlay */}
      {tourStep !== null && (
        <>
          {/* Backdrop */}
          <div className="fixed inset-0 bg-black/60 backdrop-blur-[2px] z-40 transition-opacity duration-300 pointer-events-auto" onClick={endTour} />

          {/* Tour Modal Card */}
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 pointer-events-none">
            <div className="bg-[#0E0E1B] border border-white/10 rounded-2xl max-w-md w-full p-6 shadow-2xl shadow-indigo-500/10 pointer-events-auto transform scale-100 opacity-100 transition-all duration-300 text-[#F3F4F6] relative">
              <button 
                onClick={endTour} 
                className="absolute top-4 right-4 text-slate-400 hover:text-white transition-colors cursor-pointer"
                aria-label="Close tour"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>

              <div className="flex items-center space-x-1.5 mb-3 text-[10px] text-indigo-400 font-mono uppercase tracking-wider">
                <span>Step {tourStep + 1} of 4</span>
                <span className="text-slate-600">•</span>
                <span>{tourStep === 0 ? "Introduction" : tourStep === 1 ? "Select Asset" : tourStep === 2 ? "Prompting" : "Results"}</span>
              </div>

              {tourStep === 0 && (
                <div>
                  <h3 className="text-lg font-bold bg-gradient-to-r from-white via-slate-100 to-indigo-400 bg-clip-text text-transparent mb-2">
                    Welcome to SAM3 Studio! 🚀
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed mb-6">
                    This interactive tour will guide you through segmenting and tracking objects in images, videos, and GIFs using Segment Anything Model 3 (SAM3).
                  </p>
                </div>
              )}

              {tourStep === 1 && (
                <div>
                  <h3 className="text-lg font-bold bg-gradient-to-r from-white via-slate-100 to-indigo-400 bg-clip-text text-transparent mb-2">
                    1. Select Test Asset 📸
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed mb-6">
                    Choose from our predefined samples in the sidebar. We support static images, videos, and animated GIFs! Clicking on a gallery thumbnail switches the active asset.
                  </p>
                </div>
              )}

              {tourStep === 2 && (
                <div>
                  <h3 className="text-lg font-bold bg-gradient-to-r from-white via-slate-100 to-indigo-400 bg-clip-text text-transparent mb-2">
                    2. Add Prompts & Segment 💡
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed mb-6">
                    For images, select **Text Prompt** (input keywords) or **Click Prompt** (place green/red dots on the image). For videos and GIFs, type a text prompt to propagate mask tracking across all frames, then click **Run Inference**.
                  </p>
                </div>
              )}

              {tourStep === 3 && (
                <div>
                  <h3 className="text-lg font-bold bg-gradient-to-r from-white via-slate-100 to-indigo-400 bg-clip-text text-transparent mb-2">
                    3. Segmented Output 🎯
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed mb-6">
                    View the real-time segmented output here. For videos and GIFs, this will play the processed segment animation showing SAM3 tracking in action!
                  </p>
                </div>
              )}

              <div className="flex justify-between items-center border-t border-white/5 pt-4">
                <div className="flex space-x-1.5">
                  {[0, 1, 2, 3].map((step) => (
                    <button
                      key={step}
                      onClick={() => setTourStep(step)}
                      className={`w-1.5 h-1.5 rounded-full transition-all cursor-pointer ${
                        tourStep === step ? "bg-indigo-500 w-3" : "bg-slate-700 hover:bg-slate-600"
                      }`}
                    />
                  ))}
                </div>

                <div className="flex space-x-2">
                  <button
                    onClick={endTour}
                    className="px-3 py-1.5 text-xs text-slate-400 hover:text-white transition-colors cursor-pointer"
                  >
                    Skip
                  </button>
                  {tourStep > 0 && (
                    <button
                      onClick={() => setTourStep(tourStep - 1)}
                      className="px-3 py-1.5 text-xs bg-white/5 border border-white/5 hover:border-white/10 text-white rounded-lg transition-all cursor-pointer"
                    >
                      Back
                    </button>
                  )}
                  {tourStep < 3 ? (
                    <button
                      onClick={() => setTourStep(tourStep + 1)}
                      className="px-4 py-1.5 text-xs bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-all font-medium shadow-md shadow-indigo-600/10 cursor-pointer"
                    >
                      Next
                    </button>
                  ) : (
                    <button
                      onClick={endTour}
                      className="px-4 py-1.5 text-xs bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg transition-all font-medium shadow-md shadow-emerald-600/10 cursor-pointer"
                    >
                      Finish
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
