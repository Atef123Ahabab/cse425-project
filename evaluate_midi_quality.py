#!/usr/bin/env python3
"""
Task 1 Evaluation Metrics Calculator

Computes:
- Pitch Histogram Similarity
- Rhythm Diversity Score
- Repetition Ratio
- Statistics for generated MIDI files
"""
import os
import json
import pretty_midi
import numpy as np
from collections import Counter, defaultdict
import glob

def compute_pitch_histogram(pm):
    """Compute pitch histogram (12-bin for chromatic scale)."""
    pitches = []
    for inst in pm.instruments:
        if not inst.is_drum:
            for note in inst.notes:
                pitches.append(note.pitch % 12)  # Normalize to 0-11
    
    if not pitches:
        return np.zeros(12)
    
    hist = np.zeros(12)
    for pitch in pitches:
        hist[pitch] += 1
    
    return hist / hist.sum() if hist.sum() > 0 else hist

def compute_pitch_histogram_similarity(p, q):
    """Compute pitch histogram similarity: H(p,q) = sum|p_i - q_i|"""
    return np.sum(np.abs(p - q))

def compute_rhythm_diversity(pm):
    """Compute rhythm diversity score: unique_durations / total_notes"""
    durations = []
    for inst in pm.instruments:
        if not inst.is_drum:
            for note in inst.notes:
                durations.append(note.end - note.start)
    
    if not durations:
        return 0.0
    
    unique_durations = len(set(np.round(durations, 3)))  # Round to millisecond precision
    return unique_durations / len(durations)

def compute_repetition_ratio(pm):
    """Compute repetition ratio: repeated_patterns / total_patterns"""
    sequences = []
    for inst in pm.instruments:
        if not inst.is_drum:
            pitches = [note.pitch for note in inst.notes]
            # Create 3-note patterns
            for i in range(len(pitches) - 2):
                pattern = tuple(pitches[i:i+3])
                sequences.append(pattern)
    
    if not sequences or len(sequences) < 3:
        return 0.0
    
    pattern_counts = Counter(sequences)
    repeated = sum(1 for count in pattern_counts.values() if count > 1)
    return repeated / len(pattern_counts) if pattern_counts else 0.0

def analyze_midi(midi_path):
    """Extract detailed statistics from MIDI file."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except:
        return None
    
    stats = {
        'path': midi_path,
        'filename': os.path.basename(midi_path),
        'duration': pm.get_end_time(),
        'num_instruments': len([i for i in pm.instruments if not i.is_drum]),
        'num_notes': sum(len(inst.notes) for inst in pm.instruments if not inst.is_drum),
        'pitch_range': None,
        'pitch_histogram': compute_pitch_histogram(pm),
        'rhythm_diversity': compute_rhythm_diversity(pm),
        'repetition_ratio': compute_repetition_ratio(pm),
    }
    
    # Compute pitch range
    pitches = []
    for inst in pm.instruments:
        if not inst.is_drum:
            pitches.extend([note.pitch for note in inst.notes])
    
    if pitches:
        stats['pitch_range'] = (min(pitches), max(pitches), max(pitches) - min(pitches))
    
    return stats

def main():
    # Paths to generated MIDI files
    base_output = "/Users/atefahabab/Desktop/cse425 project/outputs/maestro_task1_optimized"
    midi_dir = os.path.join(base_output, "generated_samples")
    
    if not os.path.exists(midi_dir):
        print(f"Error: MIDI directory not found: {midi_dir}")
        return
    
    midi_files = sorted(glob.glob(os.path.join(midi_dir, "*.mid")))
    
    if not midi_files:
        print(f"No MIDI files found in {midi_dir}")
        return
    
    print(f"Analyzing {len(midi_files)} generated MIDI files...\n")
    
    # Analyze all files
    all_stats = []
    for midi_path in midi_files:
        print(f"Processing: {os.path.basename(midi_path)}")
        stats = analyze_midi(midi_path)
        if stats:
            all_stats.append(stats)
            print(f"  ✓ Duration: {stats['duration']:.2f}s | Notes: {stats['num_notes']} | "
                  f"Rhythm Diversity: {stats['rhythm_diversity']:.3f} | "
                  f"Repetition: {stats['repetition_ratio']:.3f}")
    
    # Compute aggregate statistics
    print("\n" + "="*80)
    print("TASK 1 EVALUATION METRICS - AGGREGATE STATISTICS")
    print("="*80)
    
    if all_stats:
        avg_duration = np.mean([s['duration'] for s in all_stats])
        avg_notes = np.mean([s['num_notes'] for s in all_stats])
        avg_rhythm_div = np.mean([s['rhythm_diversity'] for s in all_stats])
        avg_repetition = np.mean([s['repetition_ratio'] for s in all_stats])
        avg_pitch_hist = np.mean([s['pitch_histogram'] for s in all_stats], axis=0)
        
        print(f"\nDuration Statistics:")
        print(f"  Average: {avg_duration:.2f} seconds")
        print(f"  Range: {min([s['duration'] for s in all_stats]):.2f}s - {max([s['duration'] for s in all_stats]):.2f}s")
        
        print(f"\nNote Statistics:")
        print(f"  Average notes per file: {avg_notes:.1f}")
        print(f"  Range: {min([s['num_notes'] for s in all_stats])} - {max([s['num_notes'] for s in all_stats])}")
        
        print(f"\nRhythm Diversity Score (D_rhythm):")
        print(f"  Average: {avg_rhythm_div:.4f}")
        print(f"  Range: {min([s['rhythm_diversity'] for s in all_stats]):.4f} - {max([s['rhythm_diversity'] for s in all_stats]):.4f}")
        print(f"  Interpretation: Higher = more varied note durations")
        
        print(f"\nRepetition Ratio (R):")
        print(f"  Average: {avg_repetition:.4f}")
        print(f"  Range: {min([s['repetition_ratio'] for s in all_stats]):.4f} - {max([s['repetition_ratio'] for s in all_stats]):.4f}")
        print(f"  Interpretation: Lower = less repetitive patterns (more creative)")
        
        print(f"\nPitch Histogram (Chromatic Distribution):")
        pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for i, (name, val) in enumerate(zip(pitch_names, avg_pitch_hist)):
            print(f"  {name:3s}: {val:.4f}")
        
        # Pitch histogram similarity between pairs
        print(f"\nPitch Histogram Similarity (H(p,q)) - Between Files:")
        similarities = []
        for i in range(len(all_stats)):
            for j in range(i+1, len(all_stats)):
                sim = compute_pitch_histogram_similarity(
                    all_stats[i]['pitch_histogram'],
                    all_stats[j]['pitch_histogram']
                )
                similarities.append(sim)
                print(f"  {all_stats[i]['filename']} vs {all_stats[j]['filename']}: {sim:.4f}")
        
        if similarities:
            print(f"\nAverage Pitch Similarity: {np.mean(similarities):.4f}")
            print(f"  (Lower = more diverse generated styles)")
        
        # Summary metrics
        print("\n" + "="*80)
        print("QUALITY ASSESSMENT")
        print("="*80)
        
        print(f"\n✓ Rhythmic Variety: {'Good' if avg_rhythm_div > 0.3 else 'Moderate' if avg_rhythm_div > 0.1 else 'Low'}")
        print(f"  Score: {avg_rhythm_div:.4f} (target: > 0.3)")
        
        print(f"\n✓ Creativity (Low Repetition): {'High' if avg_repetition < 0.3 else 'Medium' if avg_repetition < 0.5 else 'High repetition'}")
        print(f"  Score: {avg_repetition:.4f} (lower is better)")
        
        print(f"\n✓ Diversity (Pitch Variety): {'Good' if np.mean(similarities) > 0.15 else 'Moderate'}")
        print(f"  Score: {np.mean(similarities):.4f} (higher = more diverse)")
        
        # Save results
        results = {
            'num_files': len(all_stats),
            'files_analyzed': [s['filename'] for s in all_stats],
            'average_duration': float(avg_duration),
            'average_notes': float(avg_notes),
            'rhythm_diversity': {
                'average': float(avg_rhythm_div),
                'min': float(min([s['rhythm_diversity'] for s in all_stats])),
                'max': float(max([s['rhythm_diversity'] for s in all_stats])),
            },
            'repetition_ratio': {
                'average': float(avg_repetition),
                'min': float(min([s['repetition_ratio'] for s in all_stats])),
                'max': float(max([s['repetition_ratio'] for s in all_stats])),
            },
            'pitch_similarities': {
                'average': float(np.mean(similarities)) if similarities else 0.0,
                'pairwise': [float(s) for s in similarities],
            },
            'pitch_histogram_average': avg_pitch_hist.tolist(),
        }
        
        output_path = os.path.join(base_output, "evaluation_metrics.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Metrics saved to: {output_path}")

if __name__ == "__main__":
    main()
