import torch

PATH = "policy_epoch1.pt"

ckpt = torch.load(PATH, map_location="cpu")

print("="*50)
print("TYPE:", type(ckpt))
print("="*50)

# dict / OrderedDict
if isinstance(ckpt, dict):
    print("\nKeys:\n")
    for k in ckpt.keys():
        print(k)

    print("\nShapes:\n")
    for k, v in ckpt.items():
        if hasattr(v, "shape"):
            print(f"{k:20s} {tuple(v.shape)}")
        else:
            print(f"{k:20s} {type(v)}")

    print("\nFirst tensor preview:\n")
    for k, v in ckpt.items():
        if hasattr(v, "shape"):
            print(k)
            print(v[:5])
            break

else:
    print("Not dict — printing object:")
    print(ckpt)