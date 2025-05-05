
import React, { createContext, useContext, useState, ReactNode } from 'react';
import { PatientData, DiagnosisResult, ChatMessage } from '../services/api';

interface DiagnosisContextType {
  xrayFile: File | null;
  patientData: PatientData | null;
  diagnosisResults: DiagnosisResult[];
  chatHistory: ChatMessage[];
  sessionId: string | null;
  loading: boolean;
  setXrayFile: (file: File | null) => void;
  setPatientData: (data: PatientData | null) => void;
  setDiagnosisResults: (results: DiagnosisResult[]) => void;
  addChatMessage: (message: ChatMessage) => void;
  setSessionId: (id: string | null) => void;
  setLoading: (isLoading: boolean) => void;
  resetAll: () => void;
}

const DiagnosisContext = createContext<DiagnosisContextType | undefined>(undefined);

export const DiagnosisProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [xrayFile, setXrayFile] = useState<File | null>(null);
  const [patientData, setPatientData] = useState<PatientData | null>(null);
  const [diagnosisResults, setDiagnosisResults] = useState<DiagnosisResult[]>([]);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const addChatMessage = (message: ChatMessage) => {
    setChatHistory((prev) => [...prev, message]);
  };

  const resetAll = () => {
    setXrayFile(null);
    setPatientData(null);
    setDiagnosisResults([]);
    setChatHistory([]);
    setSessionId(null);
    setLoading(false);
  };

  return (
    <DiagnosisContext.Provider
      value={{
        xrayFile,
        patientData,
        diagnosisResults,
        chatHistory,
        sessionId,
        loading,
        setXrayFile,
        setPatientData,
        setDiagnosisResults,
        addChatMessage,
        setSessionId,
        setLoading,
        resetAll,
      }}
    >
      {children}
    </DiagnosisContext.Provider>
  );
};

export const useDiagnosis = (): DiagnosisContextType => {
  const context = useContext(DiagnosisContext);
  if (!context) {
    throw new Error('useDiagnosis must be used within a DiagnosisProvider');
  }
  return context;
};
