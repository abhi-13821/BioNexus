"""
Phase 7 Test Script
Tests all Phase 7 enhancements
"""

import sys
sys.path.insert(0, 'C:/projects/BioNexus')

print("=" * 60)
print("🧪 PHASE 7 - TESTING PRE-TRAINED MODELS")
print("=" * 60)

# Test 7.2: Biomedical Embeddings
print("\n📚 TEST 7.2: Biomedical Embeddings")
print("-" * 40)

try:
    from embeddings.biomedical_embeddings import BiomedicalEmbeddingGenerator
    generator = BiomedicalEmbeddingGenerator()
    embedding = generator.generate_embedding("EGFR mutation in lung cancer")
    print(f"✅ Embedding generated: {len(embedding)} dimensions")
    print(f"   First 5 values: {embedding[:5]}")
except Exception as e:
    print(f"❌ Test 7.2 failed: {e}")

# Test 7.3: Biomedical NER
print("\n🔬 TEST 7.3: Biomedical NER")
print("-" * 40)

try:
    from knowledge_graph.biomedical_ner import BiomedicalNER
    ner = BiomedicalNER()
    
    # Test texts
    test_texts = [
        "EGFR mutations cause resistance to aspirin in lung cancer",
        "BRCA1 and BRCA2 are associated with breast cancer",
        "The drug metformin is used for type 2 diabetes",
        "TP53 mutations are found in many cancers"
    ]
    
    print("Testing Biomedical NER on multiple texts:")
    for text in test_texts:
        entities = ner.extract_entities(text)
        print(f"\n  Text: '{text}'")
        found = False
        for key, values in entities.items():
            if values:
                print(f"    {key}: {values}")
                found = True
        if not found:
            print("    (No entities found)")
            
except Exception as e:
    print(f"❌ Test 7.3 failed: {e}")

# Test 7.4: Advanced Molecular Properties
print("\n🧪 TEST 7.4: Advanced Molecular Properties")
print("-" * 40)

try:
    from drug_discovery.advanced_properties import AdvancedMolecularProperties
    prop = AdvancedMolecularProperties()
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    result = prop.predict_properties(smiles)
    print(f"✅ Properties predicted for SMILES: {smiles}")
    for key, value in result.items():
        print(f"   - {key}: {value}")
except Exception as e:
    print(f"❌ Test 7.4 failed: {e}")

print("\n" + "=" * 60)
print("✅ PHASE 7 TEST COMPLETED")
print("=" * 60)
