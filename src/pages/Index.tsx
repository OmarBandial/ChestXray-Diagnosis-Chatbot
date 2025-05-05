
import React from 'react';
import Header from '@/components/Header';
import TabContent from '@/components/TabContent';
import { DiagnosisProvider } from '@/contexts/DiagnosisContext';

const Index = () => {
  return (
    <DiagnosisProvider>
      <div className="min-h-screen bg-gray-50 flex flex-col">
        <Header />
        <main className="flex-1 container mx-auto px-4 py-6">
          <TabContent />
        </main>
        <footer className="py-4 px-6 bg-white border-t text-center text-sm text-gray-500">
          © 2025 MedVision AI - Medical Diagnosis Assistant
        </footer>
      </div>
    </DiagnosisProvider>
  );
};

export default Index;
