import os
import PyPDF2
import tempfile
import json
import speech_recognition as sr
from gtts import gTTS
import pygame
from openai import OpenAI
import re

# ---------------- CONFIG ----------------
client = OpenAI(api_key="api key")  # safer than hardcoding
MODEL = "gpt-4o-mini"

# ---------------- PDF PARSING ----------------
def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text.strip()

def generate_json_from_model(prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a strict JSON generator."},
                {"role": "user", "content": prompt}
            ]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print("⚠ Error parsing model output:", e)
        return {}

def preprocess_text_for_tts(text):
    replacements = {
        r"\bml\b": "Machine Learning",
        r"\bai\b": "Artificial Intelligence",
        r"\bnlp\b": "Natural Language Processing",
        r"\bcv\b": "Computer Vision",
    }
    for pattern, full in replacements.items():
        text = re.sub(pattern, full, text, flags=re.IGNORECASE)
    return text

# ---------------- TTS (using pygame) ----------------
def speak_text(text, use_openai=True, voice="nova"):
    try:
        text = preprocess_text_for_tts(text)
        print(f"\nInterviewer (speaking): {text}")

        if use_openai:
            response = client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=text
            )
            tmp_file = tempfile.mktemp(suffix=".mp3")
            with open(tmp_file, "wb") as f:
                f.write(response.content)
        else:
            tts = gTTS(text=text, lang="en")
            tmp_file = tempfile.mktemp(suffix=".mp3")
            tts.save(tmp_file)

        pygame.mixer.init()
        pygame.mixer.music.load(tmp_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()
        os.remove(tmp_file)

    except Exception as e:
        print("⚠ TTS error:", e)




# ---------------- SPEECH INPUT ----------------
def listen_voice(timeout=5):
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.0          # reasonable pause
    recognizer.non_speaking_duration = 0.3    # must be <= pause_threshold

    with sr.Microphone() as source:
        print("\n🎤 Listening... (say 'exit' to quit)")
        recognizer.adjust_for_ambient_noise(source, duration=0.3)

        try:
            audio = recognizer.listen(source, timeout=timeout)
            text = recognizer.recognize_google(audio)
            print(f"You (said): {text}")
            return text
        except sr.WaitTimeoutError:
            print("⚠ No speech detected (timeout).")
            return ""
        except sr.UnknownValueError:
            print("⚠ Could not understand audio.")
            return ""
        except Exception as e:
            print("⚠ Speech recognition error:", e)
            return ""





# ---------------- RESUME PARSE ----------------
def parse_resume(pdf_file):
    resume_text = extract_text_from_pdf(pdf_file)
    parse_prompt = f"""
    Extract the following information from this resume text.
    Return valid JSON only:

    {{
      "Name": "...",
      "Skills": ["..."],
      "Experience": "...",
      "Education": "..."
    }}

    Resume:
    {resume_text}
    """
    return generate_json_from_model(parse_prompt)

# ---------------- ATS ANALYSIS ----------------
def ats_analysis(profile, role):
    ats_prompt = f"""
    Act as an Applicant Tracking System (ATS).
    Analyze this candidate profile for the role '{role}'.

    Candidate Profile:
    {json.dumps(profile, indent=2)}

    Return JSON:
    {{
      "match_score": number (0-100),
      "summary": "...",
      "missing_skills": ["..."]
    }}
    """
    return generate_json_from_model(ats_prompt)

# ---------------- SUGGEST ROLES ----------------
def suggest_roles(profile, role, missing_skills):
    alt_prompt = f"""
    Candidate applied for "{role}" but is missing skills: {', '.join(missing_skills)}.
    Based on their profile, suggest 2-3 alternate roles for which the candidate will surely have a ATS score of more than 65.

    Candidate Profile:
    {json.dumps(profile, indent=2)}

    Return JSON:
    {{
      "SuggestedRoles": ["role1", "role2"]
    }}
    """
    return generate_json_from_model(alt_prompt)

def generate_feedback(profile, answers, role, output_file="interview_feedback.txt"):
    feedback_prompt = f"""
    You are an interviewer providing detailed feedback after a mock interview.

    Candidate Profile:
    {json.dumps(profile, indent=2)}

    Role Applied: if the roles: {role}

    Candidate Answers:
    {json.dumps(answers, indent=2)}

    Please return feedback in this structured format:
    Final Score: (0-100)

    Strengths:
    - Point 1
    - Point 2

    Areas to Improve:
    - Point 1
    - Point 2
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": feedback_prompt}]
        )
        feedback_text = response.choices[0].message.content.strip()

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(feedback_text)

        print(f"\n✅ Feedback generated and saved to {output_file}\n")
        return feedback_text

    except Exception as e:
        print("⚠ Error generating feedback:", e)
        return None


def decide_interview_sections(profile, role):
    prompt = f"""
    Based on the following candidate profile and the applied role "{role}",
    divide the interview into 2-4 logical sections. 
    Each section should focus on either technical skills, behavioral aspects, 
    or domain-specific knowledge. Allocate 2-3 questions per section.

    Candidate Profile:
    {json.dumps(profile, indent=2)}

    Return JSON strictly in this format:
    {{
      "sections": {{
        "SectionName1": number_of_questions,
        "SectionName2": number_of_questions
      }}
    }}
    """
    return generate_json_from_model(prompt)


# ---------------- INTERVIEW LOOP ----------------
def interview_loop(profile, role):
    answers = []
    asked_questions = set()

    # Auto decide sections
    section_plan = decide_interview_sections(profile, role)
    INTERVIEW_SECTIONS = section_plan.get("sections", {})
    section_counts = {s: 0 for s in INTERVIEW_SECTIONS}
    current_section = list(INTERVIEW_SECTIONS.keys())[0]

    speak_text(f"We will structure your interview into the following sections: {', '.join(INTERVIEW_SECTIONS.keys())}.")

    while True:
        # Check if all sections finished
        if all(section_counts[s] >= INTERVIEW_SECTIONS[s] for s in INTERVIEW_SECTIONS):
            speak_text("That concludes our interview. Thank you for your time.")
            feedback = generate_feedback(profile, answers, role)
            print("\n📋 Interview Feedback:\n", feedback)
            speak_text("Here is your feedback. Please check the saved file for details.")
            break

        # Ask question
        q_prompt = f"""
        Candidate Role: {role}
        Candidate Profile: {json.dumps(profile, indent=2)}

        Current Section: {current_section}
        Already Asked Questions: {list(asked_questions)}

        Generate ONE new question for this section.
        Do not repeat any previous questions.
        """
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": q_prompt}]
        )
        question = response.choices[0].message.content.strip()

        # Skip if repeated
        if question in asked_questions:
            continue

        asked_questions.add(question)
        section_counts[current_section] += 1
        speak_text(question)

        # Listen for answer
        answer = listen_voice()
        if any(word in answer.lower() for word in ["exit", "quit", "stop", "thanks", "thank you"]):
            speak_text("You're welcome! It was a pleasure interviewing you.")
            feedback = generate_feedback(profile, answers, role)
            print("\n📋 Interview Feedback:\n", feedback)
            speak_text("Here is your feedback. Please check the saved file for details.")
            break

        answers.append({ "section": current_section, "question": question, "answer": answer })
        # --- Generate per-question feedback ---
        feedback_prompt = f"""
        Candidate Role: {role}
        Candidate Profile: {json.dumps(profile, indent=2)}

        Interview Question: "{question}"
        Candidate Answer: "{answer}"

        Give short constructive feedback (2-3 sentences).
        Focus on correctness, clarity, and improvements.
        """
        fb_response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": feedback_prompt}]
        )
        short_feedback = fb_response.choices[0].message.content.strip()

        print(f"\n💡 Feedback on your answer:\n{short_feedback}\n")
        speak_text(f"Here’s some feedback on your answer: {short_feedback}")

        # Move to next section if done
        if section_counts[current_section] >= INTERVIEW_SECTIONS[current_section]:
            remaining = [s for s in INTERVIEW_SECTIONS if section_counts[s] < INTERVIEW_SECTIONS[s]]
            if remaining:
                current_section = remaining[0]
# ---------------- MAIN ----------------
def main():
    pdf_path = input("Enter path to your resume PDF: ").strip()
    profile = parse_resume(pdf_path)
    print("\n📄 Resume Parsed:", json.dumps(profile, indent=2))

    # --- Ask candidate for role ---
    speak_text(f"Hello {profile.get('Name', 'Candidate')}! Please tell me the role you are applying for.")
    role = listen_voice()
    print(f"\n🎯 Candidate chose role: {role}")

    # --- ATS Analysis ---
    ats_result = ats_analysis(profile, role)
    print("\n✅ ATS Result:", json.dumps(ats_result, indent=2))

    missing_skills = ats_result.get("missing_skills", [])
    alt_roles = suggest_roles(profile, role, missing_skills)
    if alt_roles:
        speak_text(f"I also found that your profile may fit roles like {', '.join(alt_roles.get('SuggestedRoles', []))}.")
        print("\n💡 Suggested Alternate Roles:", json.dumps(alt_roles, indent=2))

    # --- Start Interview ---
    if ats_result.get("match_score", 0) >= 65:
        speak_text(f"Great! Let's begin your interview for the role of {role}.")
        interview_loop(profile, role)
    else:
        speak_text("Your ATS score is low. Consider alternate roles before the interview.")

if __name__ == "__main__":
    main()
