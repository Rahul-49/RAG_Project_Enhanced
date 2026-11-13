import React from "react";
import { Routes, Route } from "react-router-dom";
import { ThemeProvider } from "./components/ThemeProvider";
import { Sidebar } from "./components/Sidebar";

import Home from "./pages/Home";
import ChatbotPage from "./pages/ChatbotPage";
import ExperiencesPage from "./pages/ExperiencesPage";
import ResumeAnalyzerPage from "./pages/ResumeAnalyzerPage";
import RoadmapPage from "./pages/RoadmapPage";
import SkillAnalyzerPage from "./pages/SkillAnalyzerPage";

export default function App() {
  return (
    <ThemeProvider>
      <div className="flex h-screen w-screen overflow-hidden bg-gray-50 dark:bg-[#050505] text-gray-900 dark:text-white transition-colors duration-300">
        <Sidebar />
        <div className="flex-1 relative flex flex-col h-full min-h-0 overflow-hidden">
          {/* Ambient Background */}
          <div className="absolute top-[-20%] left-[-10%] w-[500px] h-[500px] bg-purple-600/20 rounded-full blur-[128px] pointer-events-none" />
          <div className="absolute bottom-[-20%] right-[-10%] w-[500px] h-[500px] bg-blue-600/20 rounded-full blur-[128px] pointer-events-none" />

          <main className="flex-1 h-full w-full min-h-0 overflow-hidden relative">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/chatbot" element={<ChatbotPage />} />
              <Route path="/roadmap" element={<RoadmapPage />} />
              <Route path="/skill-analyzer" element={<SkillAnalyzerPage />} />
              <Route path="/resume-analyzer" element={<ResumeAnalyzerPage />} />
              <Route path="/experiences" element={<ExperiencesPage />} />
            </Routes>
          </main>
        </div>
      </div>
    </ThemeProvider>
  );
}
