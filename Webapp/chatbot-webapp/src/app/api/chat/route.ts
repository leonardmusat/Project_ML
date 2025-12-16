import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const { message } = await request.json();
    if (!message) {
      return NextResponse.json({ error: "Message is required" }, { status: 400 });
    }
    console.log("Received message:", message);

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
    let errorMessage = "Failed to process chat message.";
    if (err instanceof SyntaxError) {
      // This error is often thrown by request.json()
      errorMessage = "Invalid JSON in request body.";
    }
    console.error("Error in /api/chat:", err);
    return NextResponse.json(
      { error: errorMessage },
      { status: 500 }
    );
  }
}
