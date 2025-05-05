
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCaption, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Progress } from "@/components/ui/progress";
import { useDiagnosis } from '@/contexts/DiagnosisContext';
import { Button } from '@/components/ui/button';
import { AlertCircle, FileText } from 'lucide-react';

const DiagnosisResults: React.FC = () => {
  const { xrayFile, patientData, diagnosisResults, visualization } = useDiagnosis();

  if (!xrayFile || !patientData || diagnosisResults.length === 0) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle>Diagnosis Results</CardTitle>
          <CardDescription>
            Your diagnosis results will appear here after processing.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col items-center justify-center py-12">
          <AlertCircle className="h-16 w-16 text-gray-300 mb-4" />
          <h3 className="text-lg font-medium text-gray-700 mb-2">No Results Available</h3>
          <p className="text-sm text-gray-500 text-center max-w-md mb-6">
            Please upload an X-ray image and complete the patient information form to receive your diagnosis results.
          </p>
          <div className="flex gap-4">
            <Button variant="outline">Upload X-ray</Button>
            <Button>Enter Patient Info</Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.7) return 'text-green-700';
    if (confidence > 0.4) return 'text-amber-700';
    return 'text-gray-600';
  };

  const getProgressColor = (confidence: number) => {
    if (confidence > 0.7) return 'bg-green-600';
    if (confidence > 0.4) return 'bg-amber-500';
    return 'bg-gray-400';
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Diagnosis Results</CardTitle>
        <CardDescription>
          Analysis based on your X-ray image and provided information.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-gray-500">X-ray Image</h3>
            <div className="border rounded-md overflow-hidden h-72 flex items-center justify-center bg-gray-50">
              {xrayFile && (
                <img
                  src={URL.createObjectURL(xrayFile)}
                  alt="X-ray"
                  className="object-contain max-w-full max-h-full"
                />
              )}
            </div>
          </div>
          
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-gray-500">Patient Summary</h3>
            <div className="border rounded-md p-4 bg-gray-50">
              <div className="grid grid-cols-2 gap-y-2 text-sm">
                <span className="font-medium">Age:</span>
                <span>{patientData.age}</span>
                
                <span className="font-medium">Gender:</span>
                <span className="capitalize">{patientData.gender}</span>
                
                <span className="font-medium">Pain Level:</span>
                <span>{patientData.painLevel}/10</span>
              </div>
              
              <div className="mt-3">
                <span className="font-medium text-sm">Symptoms:</span>
                <div className="flex flex-wrap gap-1 mt-1">
                  {patientData.symptoms.map((symptom, index) => (
                    <span 
                      key={index} 
                      className="inline-block bg-medical-light text-medical-primary text-xs px-2 py-1 rounded-full"
                    >
                      {symptom}
                    </span>
                  ))}
                  {patientData.symptoms.length === 0 && (
                    <span className="text-xs text-gray-500">None reported</span>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {visualization && (
          <div>
            <h3 className="text-sm font-medium text-gray-500 mb-2">Diagnosis Visualization</h3>
            <div className="border rounded-md overflow-hidden bg-white p-2 flex justify-center">
              <img 
                src={`data:image/png;base64,${visualization}`}
                alt="Diagnosis Visualization"
                className="max-w-full"
              />
            </div>
          </div>
        )}
        
        <div>
          <h3 className="text-sm font-medium text-gray-500 mb-2">Diagnosis</h3>
          <Table>
            <TableCaption>AI-assisted diagnosis based on image analysis and patient data.</TableCaption>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[200px]">Condition</TableHead>
                <TableHead>Description</TableHead>
                <TableHead className="text-right w-[150px]">Confidence</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {diagnosisResults.map((result, index) => (
                <TableRow key={index}>
                  <TableCell className="font-medium">{result.ailment}</TableCell>
                  <TableCell>{result.description}</TableCell>
                  <TableCell className="text-right">
                    <div className="flex items-center justify-end space-x-2">
                      <span className={`text-sm ${getConfidenceColor(result.confidence)}`}>
                        {Math.round(result.confidence * 100)}%
                      </span>
                      <Progress 
                        value={result.confidence * 100} 
                        max={100} 
                        className={`w-20 h-2 ${getProgressColor(result.confidence)}`} 
                      />
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>

        <div className="flex justify-end space-x-4 pt-4">
          <Button variant="outline">
            <FileText className="mr-2 h-4 w-4" />
            Export Report
          </Button>
          <Button>
            Consult with AI
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default DiagnosisResults;
