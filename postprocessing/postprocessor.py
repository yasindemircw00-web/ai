import torch

def apply_labels_to_segments(segments, labels):
    """
    Kümeleme sonuçlarını segmentlere uygular.
    
    Args:
        segments (list): Ses segmentleri.
        labels (list): Küme etiketleri.
        
    Returns:
        labeled_segments (list): Etiketlenmiş segmentler.
    """
    labeled_segments = [(segment, label) for segment, label in zip(segments, labels)]
    return labeled_segments

def merge_short_segments(labeled_segments, min_duration=1.0, sample_rate=16000):
    """
    Kısa segmentleri birleştirir.
    
    Args:
        labeled_segments (list): Etiketlenmiş segmentler.
        min_duration (float): Minimum segment süresi (saniye).
        sample_rate (int): Örnekleme frekansı.
        
    Returns:
        merged_segments (list): Birleştirilmiş segmentler.
    """
    if len(labeled_segments) == 0:
        return []
    
    merged_segments = []
    current_segment = labeled_segments[0][0]
    current_label = labeled_segments[0][1]
    
    for segment, label in labeled_segments[1:]:
        if label == current_label:
            # Aynı etiketli segmentleri birleştir
            if current_segment.dim() == 2:
                current_segment = torch.cat([current_segment, segment], dim=1)
            elif current_segment.dim() == 3:
                current_segment = torch.cat([current_segment, segment], dim=2)
        else:
            # Farklı etiketli segmenti işle
            segment_duration = current_segment.shape[1] / sample_rate if current_segment.dim() == 2 else current_segment.shape[2] / sample_rate
            if segment_duration >= min_duration:
                merged_segments.append((current_segment, current_label))
            current_segment = segment
            current_label = label
    
    # Son segmenti ekle
    segment_duration = current_segment.shape[1] / sample_rate if current_segment.dim() == 2 else current_segment.shape[2] / sample_rate
    if segment_duration >= min_duration:
        merged_segments.append((current_segment, current_label))
    
    return merged_segments