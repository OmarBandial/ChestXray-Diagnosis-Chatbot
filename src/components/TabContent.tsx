
import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import XRayUpload from './tabs/XRayUpload';
import PatientForm from './tabs/PatientForm';
import DiagnosisResults from './tabs/DiagnosisResults';
import ChatWithAI from './tabs/ChatWithAI';

const TabContent: React.FC = () => {
  return (
    <div className="p-6">
      <Tabs defaultValue="upload" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="upload">Upload X-ray</TabsTrigger>
          <TabsTrigger value="patient">Patient Info</TabsTrigger>
          <TabsTrigger value="results">Results</TabsTrigger>
          <TabsTrigger value="chat">Chat with AI</TabsTrigger>
        </TabsList>
        <TabsContent value="upload">
          <XRayUpload />
        </TabsContent>
        <TabsContent value="patient">
          <PatientForm />
        </TabsContent>
        <TabsContent value="results">
          <DiagnosisResults />
        </TabsContent>
        <TabsContent value="chat">
          <ChatWithAI />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default TabContent;
