from torch.optim.lr_scheduler import StepLR

# Optimizer with differential learning rates
optimizer = optim.Adam([
    {'params': model.vision_model.parameters(), 'lr': 1e-5},
    {'params': model.language_model.parameters(), 'lr': 1e-5},
    {'params': model.fc_combined.parameters(), 'lr': 1e-4},
])

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# Training loop with fine-tuning
for epoch in range(NUM_EPOCHS):
    model.train()
    for images, encoded_texts, labels in train_loader:
        optimizer.zero_grad()
        
        images = images.to(device)
        input_ids = encoded_texts['input_ids'].squeeze(1).to(device)
        attention_mask = encoded_texts['attention_mask'].squeeze(1).to(device)
        
        outputs = model(images, input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
    
    scheduler.step()  # Update the learning rate

    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item()}')
