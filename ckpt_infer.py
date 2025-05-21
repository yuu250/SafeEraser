import os
import json
import torch
import random
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor
from transformers import LlavaForConditionalGeneration
import argparse
from peft import LoraConfig, get_peft_model
from conversation import conv_templates  

random.seed(233)

def main(args):
    file = args.eval_file
    model_path = args.model_path
    output_file = args.output_file
    tokens_text=512
    tokens_image=512
    tokens_sd=512
    tokens_harm=512


    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16
    )
    image_processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-large-patch14-336"
    )

    if args.checkpoint_path is not None:
        print("Merging Lora Weights.....")
        target_modules = r'.*language_model.*\.(up_proj|k_proj|linear_2|down_proj|v_proj|q_proj|o_proj|gate_proj|linear_1)'
        config = LoraConfig(
            r=32,
            lora_alpha=256,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)
        model.load_state_dict(
            torch.load(args.checkpoint_path),
            strict=False
        )
        model.merge_and_unload()

    model.half().to("cuda:0")
    model.eval()

    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []

    def generate_responses(prompt_text, image_tensor,
                           num_responses=1,
                           temperature=1.0,
                           top_p=0.9,
                           num_beams=1,
                          my_max_new_tokens=512):

        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        inputs = tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        inputs["pixel_values"] = image_tensor

        all_predictions = []
        for _ in range(num_responses):
            output = model.generate(
                **inputs,
                max_new_tokens=my_max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams
            )
            decoded = tokenizer.decode(
                output[0], skip_special_tokens=True
            ).strip()

            if "ASSISTANT:" in decoded:
                idx = decoded.index("ASSISTANT:")
                prediction = decoded[idx:].replace("ASSISTANT:", "", 1).strip()
            else:
                prediction = decoded


            if prediction.endswith("</s>"):
                prediction = prediction[:-len("</s>")].strip()

            all_predictions.append(prediction)

        return all_predictions

    for line in tqdm(data):
        sd_image_path = line.get("SDImage_path", None)
        if sd_image_path and os.path.exists(sd_image_path):
            sd_image = Image.open(sd_image_path).convert('RGB')
            sd_image_tensor = image_processor.preprocess(
                sd_image, return_tensors='pt'
            )['pixel_values'].half().to(model.device)
        else:
            print("sd_image_tensor is None")
            sd_image_tensor = None


        image_id_path = line.get("image_path", None)
        if image_id_path and os.path.exists(image_id_path):
            id_image = Image.open(image_id_path).convert('RGB')
            id_image_tensor = image_processor.preprocess(
                id_image, return_tensors='pt'
            )['pixel_values'].half().to(model.device)
        else:
            print("id_image_tensor is None")
            id_image_tensor = None

        output_line = {}
        for k, v in line.items():
            output_line[k] = v

        unsafe_pairs = output_line.get("unsafe_pairs", [])
        if sd_image_tensor is not None:
            for up in unsafe_pairs:
                q = up.get("question", "")

                prompt_text = f"<image>\n{q}"
                new_sd_resp = generate_responses(prompt_text, sd_image_tensor, num_responses=1,my_max_new_tokens=tokens_sd)

                if new_sd_resp:
                    up["sd_response"] = new_sd_resp[0]

        if id_image_tensor is not None:
            if "UnharmPair_text1" in output_line and isinstance(output_line["UnharmPair_text1"], dict):
                q_text1 = output_line["UnharmPair_text1"].get("Question", "")
                if q_text1.strip():
                    prompt_text1 = f"<image>\n{q_text1}"
                    new_pred_text1 = generate_responses(prompt_text1, id_image_tensor, num_responses=1,my_max_new_tokens=tokens_text)
                    if new_pred_text1:
                        output_line["UnharmPair_text1"]["Prediction"] = new_pred_text1[0]

            if "UnharmPair_text2" in output_line and isinstance(output_line["UnharmPair_text2"], dict):
                q_text2 = output_line["UnharmPair_text2"].get("Question", "")
                if q_text2.strip():
                    prompt_text2 = f"<image>\n{q_text2}"
                    new_pred_text2 = generate_responses(prompt_text2, id_image_tensor, num_responses=1,my_max_new_tokens=tokens_text)
                    if new_pred_text2:
                        output_line["UnharmPair_text2"]["Prediction"] = new_pred_text2[0]

            if "UnharmPair_image1" in output_line and isinstance(output_line["UnharmPair_image1"], dict):
                q_img1 = output_line["UnharmPair_image1"].get("Question", "")
                if q_img1.strip():
                    prompt_img1 = f"<image>\n{q_img1}"
                    new_pred_img1 = generate_responses(prompt_img1, id_image_tensor, num_responses=1,my_max_new_tokens=tokens_image)
                    if new_pred_img1:
                        output_line["UnharmPair_image1"]["Prediction"] = new_pred_img1[0]

            if "UnharmPair_image2" in output_line and isinstance(output_line["UnharmPair_image2"], dict):
                q_img2 = output_line["UnharmPair_image2"].get("Question", "")
                if q_img2.strip():
                    prompt_img2 = f"<image>\n{q_img2}"
                    new_pred_img2 = generate_responses(prompt_img2, id_image_tensor, num_responses=1,my_max_new_tokens=tokens_image)
                    if new_pred_img2:
                        output_line["UnharmPair_image2"]["Prediction"] = new_pred_img2[0]

        if id_image_tensor is not None:
            for up in unsafe_pairs:
                if "model_response" in up:
                    del up["model_response"]

                q = up.get("question", "")
                prompt_text = f"<image>\n{q}"
                model_preds = generate_responses(prompt_text, id_image_tensor, num_responses=3,my_max_new_tokens=tokens_harm)
                if len(model_preds) >= 1:
                    up["model_response1"] = model_preds[0]
                if len(model_preds) >= 2:
                    up["model_response2"] = model_preds[1]
                if len(model_preds) >= 3:
                    up["model_response3"] = model_preds[2]

        output_line["unsafe_pairs"] = unsafe_pairs

        results.append(output_line)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, required=True, help="the path to the eval file")
    parser.add_argument("--model_path", type=str, default="llava-hf/llava-1.5-7b-hf", help="model path")
    parser.add_argument("--output_file", type=str, required=True, help="path to save the output results")
    parser.add_argument("--checkpoint_path", choices=None, default=None, type=str, help="lora weights of unlearning methods")
    parser.add_argument("--loss_type", choices=["ga", "kl", "po",  "retain", "full","gd","gapd","gdpd","klpd","popd","idk","idkpd"], default="ga", type=str, help="unlearning method")
    args = parser.parse_args()
    main(args)