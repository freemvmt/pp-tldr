import torch
import whisper

# ---------- pre-processing ----------
model = whisper.load_model("tiny")
audio = whisper.load_audio("./freemvmt/samples/super.aac")
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio)
tknsr = whisper.tokenizer.get_tokenizer(multilingual=True)  # type: ignore
ground_truth = "Supercalifragilisticexpialidocious"

# ---------- baseline decode ----------
opt = whisper.DecodingOptions()
res = whisper.decode(model, mel.to(model.device), opt)
print("Baseline:", res.text)  # type: ignore
print("______")

# ---------- build target token list ----------
ids = []
ids += [tknsr.sot]
ids += [tknsr.language_token]
ids += [tknsr.transcribe]
ids += [tknsr.no_timestamps]
ids += tknsr.encode(f" {ground_truth}")
ids += [tknsr.eot]

# ---------- training set-up ----------
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
criterion = torch.nn.CrossEntropyLoss()

# ---------- single training step ----------
model.train()
tks = torch.tensor(ids).unsqueeze(0).to(model.device)
mel = whisper.log_mel_spectrogram(audio).unsqueeze(0).to(model.device)

pred = model(tokens=tks, mel=mel)
trgt = tks[:, 1:].contiguous()  # shift-right for labels
pred = pred[:, :-1, :].contiguous()  # align predictions

print("Ids Target:", trgt.squeeze().tolist())
print("Ids Output:", torch.argmax(pred, dim=-1).squeeze().tolist())
print("Txt Target:", tknsr.decode(trgt.squeeze().tolist()))
print("Txt Output:", tknsr.decode(torch.argmax(pred, dim=-1).squeeze().tolist()))

loss = criterion(pred.transpose(1, 2), trgt)
print("Loss:", loss.item())
print("______")

optimizer.zero_grad()
loss.backward()
optimizer.step()

# ---------- evaluation ----------
model.eval()
prd = model(tokens=tks, mel=mel)
prd = prd[:, :-1, :].contiguous()

print("Ids Target:", trgt.squeeze().tolist())
print("Ids Output:", torch.argmax(prd, dim=-1).squeeze().tolist())
print("Txt Target:", tknsr.decode(trgt.squeeze().tolist()))
print("Txt Output:", tknsr.decode(torch.argmax(prd, dim=-1).squeeze().tolist()))

loss = criterion(prd.transpose(1, 2), trgt)
print("Loss:", loss.item())
