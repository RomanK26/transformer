
import math
import time
from tqdm import tqdm
from torch import nn, optim
from torch.optim import Adam
from dataset import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time

# dataset = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda batch: custom_collate(batch, dataset.src_sos_idx, dataset.src_eos_idx, dataset.trg_sos_idx, dataset.trg_eos_idx, dataset.src_pad_idx, dataset.trg_pad_idx))
train_iter = DataLoader(dataset.train_data, batch_size=8, shuffle=True, collate_fn=lambda batch: custom_collate(batch, dataset.src_sos_idx, dataset.src_eos_idx, dataset.trg_sos_idx, dataset.trg_eos_idx, dataset.src_pad_idx, dataset.trg_pad_idx,dataset.src_vocab_dict,dataset.trg_vocab_dict))
test_iter = DataLoader(dataset.test_data, batch_size=8, shuffle=False, collate_fn=lambda batch: custom_collate(batch, dataset.src_sos_idx, dataset.src_eos_idx, dataset.trg_sos_idx, dataset.trg_eos_idx, dataset.src_pad_idx, dataset.trg_pad_idx,dataset.src_vocab_dict,dataset.trg_vocab_dict))
valid_iter = DataLoader(dataset.val_data, batch_size=8, shuffle=False, collate_fn=lambda batch: custom_collate(batch, dataset.src_sos_idx, dataset.src_eos_idx, dataset.trg_sos_idx, dataset.trg_eos_idx, dataset.src_pad_idx, dataset.trg_pad_idx,dataset.src_vocab_dict,dataset.trg_vocab_dict))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)


model = Transformer(src_pad_idx=dataset.src_pad_idx,
                    trg_pad_idx=dataset.trg_pad_idx,
                    trg_sos_idx=dataset.trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=dataset.enc_voc_size,
                    dec_voc_size=dataset.dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=dataset.src_pad_idx)



def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    correct_predictions = 0
    total_samples = 0

    iterator = tqdm(iterator, total=len(iterator), desc='Training')
    
    for i, batch in enumerate(iterator):
        src, trg = batch
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        # Compute accuracy
        predicted = output_reshape.argmax(dim=1)
        correct_predictions += (predicted == trg).sum().item()
        total_samples += trg.size(0)

        accuracy = correct_predictions / total_samples

        # Update the progress bar description
#         iterator.set_description(f'Training Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f}')

    return epoch_loss / len(iterator), accuracy




def evaluate(model, iterator, criterion, target_vocab, device):
    model.eval()
    epoch_loss = 0
    total_correct = 0
    total_samples = 0
    accuracies = []
    references = []  # List to store reference sentences
    hypotheses = []  # List to store predicted sentences

    iterator = tqdm(iterator, total=len(iterator), desc='Evaluating')
    
    with torch.no_grad():
        for i, (src_batch, trg_batch) in enumerate(iterator):
            src = src_batch
            trg = trg_batch
            src, trg = src.to(device), trg.to(device)

            # Forward pass
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            output_reshape = output_reshape.to(device)
            trg = trg[:, 1:].contiguous().view(-1)
            trg = trg.to(output.device)

            # Calculate loss
            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            # Calculate accuracy
            predicted = output_reshape.argmax(dim=1)
            correct = (predicted == trg).sum().item()
            total_correct += correct
            total_samples += trg.size(0)
            accuracy = total_correct / total_samples
            accuracies.append(accuracy)

            # Convert indices to words for BLEU calculation
            trg_words = [idx_to_word(sentence, target_vocab) for sentence in trg_batch.T]
            output_words = [idx_to_word(sentence, target_vocab) for sentence in output.argmax(dim=2).T]


            references.extend(trg_words)
            hypotheses.extend(output_words)

            # print(f"trg size: {trg.size()}\n")
            # print(f"output size: {output.size()}\n")

    # Calculate BLEU score
    bleu = get_bleu(hypotheses, references)
    
    return epoch_loss / len(iterator), accuracies, bleu
        
        
        


def run(total_epoch, best_loss):
    train_losses, test_losses, bleus, train_accuracies, val_accuracies = [], [], [], [],[]
    for step in range(total_epoch):
        start_time = time.time()
        train_loss,train_accuracy = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu, val_accuracy = evaluate(model, valid_iter, criterion, dataset.trg_vocab_dict, device)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        train_accuracies.append(train_accuracy)  # Append train accuracy to the list
        val_accuracies.append(val_accuracy)  # Append accuracy to the list
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            # Save the model if needed
            # torch.save(model.state_dict(), '/kaggle/working/model-{0}.pt'.format(valid_loss))

        # result_directory = '/kaggle/working/result'
        # os.makedirs(result_directory, exist_ok=True)
        
        # Save metrics to separate files
        with open('/Users/romankasichhwa/Desktop/project/transformer/saved/transformer-base/train.txt', 'w') as f:
            f.write(str(train_losses))

        with open('/Users/romankasichhwa/Desktop/project/transformer/saved/transformer-base/bleu.txt', 'w') as f:
            f.write(str(bleus))

        with open('/Users/romankasichhwa/Desktop/project/transformer/saved/transformer-base/test_loss.txt', 'w') as f:
            f.write(str(test_losses))

        with open('/Users/romankasichhwa/Desktop/project/transformer/saved/transformer-base/train_accuracies.txt', 'w') as f:
            f.write(str(train_accuracies))

        with open('/Users/romankasichhwa/Desktop/project/transformer/saved/transformer-base/val_accuracies.txt', 'w') as f:
            f.write(str(val_accuracies))

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Train Accuracy: {train_accuracies[-1]*100:.3f} %')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f} | Val Accuracy: {val_accuracies[-1]*100:.3f} %')
        print(f'\tBLEU Score: {bleus[-1][-1]:.3f}')


        
        

    torch.save(model.state_dict(), '/Users/romankasichhwa/Desktop/project/transformer/saved/model-1.pt'.format(valid_loss))

if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
    draw(mode='loss',total_epoch=epoch, live_update=True)
    draw(mode='bleu',total_epoch=epoch, live_update=True)
    draw(mode='accuracy',total_epoch=epoch, live_update=True)