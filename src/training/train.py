"""Training utilities for video classification."""
import torch

# Normal training function for single-stream models

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

# Specialized training function for temporal stream models

def train_temporal_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for rgb_frames, flow_stack, labels in loader:
        # Only use flow_stack for temporal stream
        flow_stack = flow_stack.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(flow_stack)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def validate_temporal(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for rgb_frames, flow_stack, labels in loader:
            # Only use flow_stack for temporal stream
            flow_stack = flow_stack.to(device)
            labels = labels.to(device)

            outputs = model(flow_stack)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def train_epoch_spatial(model, loader, criterion, optimizer, device):
    """Train spatial stream for one epoch (extracts only RGB from DualStreamDataset)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for rgb_frames, flow_stack, labels in loader:
        # Only use rgb_frames for spatial stream
        rgb_frames = rgb_frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(rgb_frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate_spatial(model, loader, criterion, device):
    """Validate spatial stream (extracts only RGB from DualStreamDataset)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for rgb_frames, flow_stack, labels in loader:
            # Only use rgb_frames for spatial stream
            rgb_frames = rgb_frames.to(device)
            labels = labels.to(device)

            outputs = model(rgb_frames)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc