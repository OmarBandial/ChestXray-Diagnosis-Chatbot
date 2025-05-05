
import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import XRayUpload from './tabs/XRayUpload';
import PatientForm from './tabs/PatientForm';
import DiagnosisResults from './tabs/DiagnosisResults';
import ChatWithAI from './tabs/ChatWithAI';
import { Card } from "@/components/ui/card";
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable";

const TabContent: React.FC = () => {
  return (
    <ResizablePanelGroup 
      direction="horizontal" 
      className="w-full rounded-lg border"
    >
      {/* First Box - Upload X-Ray and Patient Info */}
      <ResizablePanel defaultSize={50} minSize={30}>
        <Card className="h-full rounded-none">
          <Tabs defaultValue="upload" className="w-full h-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="upload">Upload X-ray</TabsTrigger>
              <TabsTrigger value="patient">Patient Info</TabsTrigger>
            </TabsList>
            <TabsContent value="upload" className="h-[calc(100%-45px)]">
              <XRayUpload />
            </TabsContent>
            <TabsContent value="patient" className="h-[calc(100%-45px)]">
              <PatientForm />
            </TabsContent>
          </Tabs>
        </Card>
      </ResizablePanel>
      
      {/* Resizable Handle */}
      <ResizableHandle withHandle />
      
      {/* Second Box - Results and Chat with AI */}
      <ResizablePanel defaultSize={50} minSize={30}>
        <Card className="h-full rounded-none">
          <Tabs defaultValue="results" className="w-full h-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="results">Results</TabsTrigger>
              <TabsTrigger value="chat">Chat with AI</TabsTrigger>
            </TabsList>
            <TabsContent value="results" className="h-[calc(100%-45px)]">
              <DiagnosisResults />
            </TabsContent>
            <TabsContent value="chat" className="h-[calc(100%-45px)]">
              <ChatWithAI />
            </TabsContent>
          </Tabs>
        </Card>
      </ResizablePanel>
    </ResizablePanelGroup>
  );
};

export default TabContent;
