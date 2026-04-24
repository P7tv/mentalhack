import Groq from 'groq-sdk';

const groq = new Groq({
  apiKey: import.meta.env.VITE_GROQ_API_KEY,
  dangerouslyAllowBrowser: true,
});

export async function analyzeWithGroq(text) {
  const prompt = `คุณเป็นนักจิตวิทยาผู้เชี่ยวชาญด้านสุขภาพจิต วิเคราะห์ข้อความต่อไปนี้ เพื่อเข้าใจสภาวะสุขภาพจิตของคน ให้คำแนะนำที่เห็นอกเห็นใจและมีประโยชน์

ข้อความ: "${text}"

กรุณา:
1. ระบุอารมณ์หลักที่ตรวจพบ
2. อธิบายสภาวะสุขภาพจิตที่อาจเกี่ยวข้อง
3. ให้คำแนะนำเชิงสร้างสรรค์
4. ระบุว่าหากปัญหารุนแรง ให้ติดต่อผู้เชี่ยวชาญ

ตอบเป็นภาษาไทยเท่านั้น`;

  const message = await groq.messages.create({
    messages: [
      {
        role: 'user',
        content: prompt,
      },
    ],
    model: 'mixtral-8x7b-32768',
    max_tokens: 1024,
  });

  return message.content[0].text;
}
