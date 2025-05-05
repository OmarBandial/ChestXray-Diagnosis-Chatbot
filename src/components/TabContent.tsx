
import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import XRayUpload from './tabs/XRayUpload';
import PatientForm from './tabs/PatientForm';
import DiagnosisResults from './tabs/DiagnosisResults';
import ChatWithAI from './tabs/ChatWithAI';
import { Card } from "@/components/ui/card";

const TabContent: React.FC = () => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-full">
      {/* First Box - Upload X-Ray and Patient Info */}
      <Card className="h-full shadow-md">
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
      
      {/* Second Box - Results and Chat with AI */}
      <Card className="h-full shadow-md">
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
    </div>
  );
};

export default TabContent;
