"use client";

interface ModelSelectorProps {
  selectedModel: string;
  setSelectedModel: (model: string) => void;
}

export default function ModelSelector({ selectedModel, setSelectedModel }: ModelSelectorProps) {
  const models = ["SVM", "Logistic Regression", "SGD", "All Models"];

  return (
    <div className="flex items-center space-x-2">
      <label htmlFor="model-selector" className="text-white">
        Choose a model:
      </label>
      <select
        id="model-selector"
        className="p-2 rounded-lg bg-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        value={selectedModel}
        onChange={(e) => setSelectedModel(e.target.value)}
      >
        {models.map((model) => (
          <option key={model} value={model}>
            {model}
          </option>
        ))}
      </select>
    </div>
  );
}
