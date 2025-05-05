
import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { useDiagnosis } from '@/contexts/DiagnosisContext';
import { submitPatientData, PatientData } from '@/services/api';
import { Check, User } from 'lucide-react';

const PatientForm: React.FC = () => {
  const { sessionId, setPatientData, setDiagnosisResults, setLoading } = useDiagnosis();
  const { toast } = useToast();
  
  const [formData, setFormData] = useState<PatientData>({
    age: 0,
    gender: '',
    symptoms: [],
    painLevel: 0,
    medicalHistory: '',
    medications: []
  });

  const [symptomInput, setSymptomInput] = useState('');
  const [medicationInput, setMedicationInput] = useState('');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleGenderChange = (value: string) => {
    setFormData(prev => ({ ...prev, gender: value }));
  };

  const handlePainLevelChange = (value: number[]) => {
    setFormData(prev => ({ ...prev, painLevel: value[0] }));
  };

  const addSymptom = () => {
    if (symptomInput && !formData.symptoms.includes(symptomInput)) {
      setFormData(prev => ({
        ...prev,
        symptoms: [...prev.symptoms, symptomInput]
      }));
      setSymptomInput('');
    }
  };

  const removeSymptom = (symptom: string) => {
    setFormData(prev => ({
      ...prev,
      symptoms: prev.symptoms.filter(s => s !== symptom)
    }));
  };

  const addMedication = () => {
    if (medicationInput && !formData.medications.includes(medicationInput)) {
      setFormData(prev => ({
        ...prev,
        medications: [...prev.medications, medicationInput]
      }));
      setMedicationInput('');
    }
  };

  const removeMedication = (medication: string) => {
    setFormData(prev => ({
      ...prev,
      medications: prev.medications.filter(m => m !== medication)
    }));
  };

  const handleSubmit = async () => {
    if (!sessionId) {
      toast({
        title: "No session found",
        description: "Please upload an X-ray image first.",
        variant: "destructive",
      });
      return;
    }

    if (!formData.age || !formData.gender) {
      toast({
        title: "Missing information",
        description: "Please provide age and gender information.",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);
    try {
      // In a real app, this would send data to the backend
      // const response = await submitPatientData({...formData, sessionId});
      
      // Store the patient data
      setPatientData(formData);
      
      // Mock diagnosis results for demo purposes
      const mockResults = [
        {
          ailment: "Pneumonia",
          confidence: 0.87,
          description: "Inflammation of the lungs caused by infection."
        },
        {
          ailment: "Pleural Effusion",
          confidence: 0.45,
          description: "Buildup of fluid between the lungs and chest cavity."
        },
        {
          ailment: "Atelectasis",
          confidence: 0.32,
          description: "Collapse or closure of a lung resulting in reduced or absent gas exchange."
        }
      ];
      
      setDiagnosisResults(mockResults);
      
      toast({
        title: "Patient data submitted",
        description: "Your information has been processed. View the results tab for diagnosis.",
      });
    } catch (error) {
      toast({
        title: "Submission failed",
        description: "There was a problem submitting your information. Please try again.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Patient Information</CardTitle>
        <CardDescription>
          Please enter your information to help with the diagnosis.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <Label htmlFor="age">Age</Label>
            <Input
              id="age"
              name="age"
              type="number"
              placeholder="Enter your age"
              value={formData.age || ''}
              onChange={handleChange}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="gender">Gender</Label>
            <RadioGroup
              name="gender"
              value={formData.gender}
              onValueChange={handleGenderChange}
              className="flex space-x-4"
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="male" id="male" />
                <Label htmlFor="male">Male</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="female" id="female" />
                <Label htmlFor="female">Female</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="other" id="other" />
                <Label htmlFor="other">Other</Label>
              </div>
            </RadioGroup>
          </div>
        </div>

        <div className="space-y-2">
          <Label htmlFor="symptoms">Symptoms</Label>
          <div className="flex space-x-2">
            <Input
              id="symptomInput"
              placeholder="Enter symptom (e.g., cough, fever)"
              value={symptomInput}
              onChange={(e) => setSymptomInput(e.target.value)}
            />
            <Button type="button" onClick={addSymptom} variant="outline">
              Add
            </Button>
          </div>
          <div className="flex flex-wrap gap-2 mt-2">
            {formData.symptoms.map((symptom, index) => (
              <span
                key={index}
                className="bg-medical-light text-medical-primary px-3 py-1 rounded-full text-sm flex items-center"
              >
                {symptom}
                <button
                  type="button"
                  className="ml-2 text-medical-primary hover:text-medical-dark"
                  onClick={() => removeSymptom(symptom)}
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        </div>

        <div className="space-y-2">
          <Label htmlFor="painLevel">Pain Level (0-10)</Label>
          <div className="flex items-center space-x-4">
            <Slider
              id="painLevel"
              min={0}
              max={10}
              step={1}
              value={[formData.painLevel]}
              onValueChange={handlePainLevelChange}
              className="flex-1"
            />
            <span className="w-8 text-center">{formData.painLevel}</span>
          </div>
        </div>

        <div className="space-y-2">
          <Label htmlFor="medicalHistory">Medical History</Label>
          <Textarea
            id="medicalHistory"
            name="medicalHistory"
            placeholder="Enter any relevant medical history"
            value={formData.medicalHistory}
            onChange={handleChange}
            className="min-h-[100px]"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="medications">Current Medications</Label>
          <div className="flex space-x-2">
            <Input
              id="medicationInput"
              placeholder="Enter medication name"
              value={medicationInput}
              onChange={(e) => setMedicationInput(e.target.value)}
            />
            <Button type="button" onClick={addMedication} variant="outline">
              Add
            </Button>
          </div>
          <div className="flex flex-wrap gap-2 mt-2">
            {formData.medications.map((medication, index) => (
              <span
                key={index}
                className="bg-gray-100 px-3 py-1 rounded-full text-sm flex items-center"
              >
                {medication}
                <button
                  type="button"
                  className="ml-2 text-gray-500 hover:text-gray-700"
                  onClick={() => removeMedication(medication)}
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex justify-end">
        <Button type="button" onClick={handleSubmit}>
          <Check className="mr-2 h-4 w-4" />
          Submit Patient Information
        </Button>
      </CardFooter>
    </Card>
  );
};

export default PatientForm;
