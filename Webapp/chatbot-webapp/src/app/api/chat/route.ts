import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const { message } = await request.json();
  console.log("Received message:", message);

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      // This matches the Pydantic model: Requirement(requirement_text: str)
      body: JSON.stringify({ requirement_text: message }),
    });

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const data = await response.json();
    // data = { requirement: "...", predicted_class: "..." }

    // Just forward it to the frontend in a nice shape
    return NextResponse.json({
      requirement: data.requirement,
      predicted_class: data.predicted_class,
    });
  } catch (err) {
    console.error("Error calling FastAPI:", err);
    return NextResponse.json(
      { error: "Failed to call prediction service" },
      { status: 500 }
    );
  }
}
