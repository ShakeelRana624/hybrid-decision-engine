"""
Comprehensive Test Suite for Hybrid Decision Engine
Tests all threat levels from NORMAL to CRITICAL
"""

import unittest
import time
import numpy as np
from hybrid_decision_engine import DecisionEngine, Threat, State

class TestHybridDecisionEngine(unittest.TestCase):
    
    def setUp(self):
        """Setup test engine before each test"""
        self.engine = DecisionEngine()
        
    def create_detection(self, person_conf=0.9, gun_conf=0.0, knife_conf=0.0, 
                        fight_conf=0.0, person_id=1, meta=None):
        """Helper to create detection dictionary"""
        return {
            "id": person_id,
            "bbox": [100, 100, 80, 200],
            "person_conf": person_conf,
            "gun_conf": gun_conf,
            "knife_conf": knife_conf,
            "fight_conf": fight_conf,
            "meta": meta or {},
            "timestamp": time.time()
        }
    
    def test_normal_state(self):
        """Test NORMAL state - no threats detected"""
        detection = self.create_detection()
        result = self.engine.process(detection)
        
        self.assertEqual(result['state'], State.NORMAL)
        self.assertLess(result['threat_score'], 1.6)  # Below medium threshold
        self.assertEqual(result['action'], 'NO_ACTION')
        print(f"✓ NORMAL state: score={result['threat_score']}, action={result['action']}")
    
    def test_suspicious_state(self):
        """Test SUSPICIOUS state - low level threats"""
        # Simulate loitering behavior
        detection = self.create_detection(meta={"loitering": True})
        
        # Process multiple frames to trigger suspicious state
        for i in range(15):
            result = self.engine.process(detection)
            if i >= 10:  # After suspicious duration
                self.assertEqual(result['state'], State.SUSPICIOUS)
        
        self.assertGreaterEqual(result['threat_score'], 1.6)
        self.assertIn('LOG', result['action'])
        print(f"✓ SUSPICIOUS state: score={result['threat_score']}, action={result['action']}")
    
    def test_knife_detection_armed_state(self):
        """Test ARMED state - knife detection"""
        detection = self.create_detection(knife_conf=0.6)  # Above knife threshold
        
        # Process multiple frames to escalate - need more frames for ARMED state
        for i in range(12):
            result = self.engine.process(detection)
        
        self.assertEqual(result['state'], State.ARMED)
        self.assertGreaterEqual(result['threat_score'], 2.0)
        self.assertIn('SAVE_EVIDENCE', result['action'])
        self.assertIn('LOCAL_ALARM', result['action'])
        print(f"✓ ARMED state (knife): score={result['threat_score']}, action={result['action']}")
    
    def test_gun_detection_violent_state(self):
        """Test VIOLENT state - gun detection"""
        detection = self.create_detection(gun_conf=0.8)  # Above gun threshold
        
        # Process frames to escalate to violent
        for i in range(8):
            result = self.engine.process(detection)
        
        self.assertEqual(result['state'], State.VIOLENT)
        self.assertGreaterEqual(result['threat_score'], 2.8)
        self.assertIn('SAVE_EVIDENCE', result['action'])
        self.assertIn('LOCAL_ALARM', result['action'])
        self.assertIn('NOTIFY_OPERATOR', result['action'])
        print(f"✓ VIOLENT state (gun): score={result['threat_score']}, action={result['action']}")
    
    def test_fight_detection_critical_state(self):
        """Test CRITICAL state - fight detection"""
        detection = self.create_detection(fight_conf=0.9)  # Above fight threshold
        
        # Process frames to escalate to critical - need to build up through states
        # First escalate to violent with some threat
        for i in range(5):
            detection['fight_conf'] = 0.7  # Start with high but not critical
            result = self.engine.process(detection)
        
        # Then push to critical
        detection['fight_conf'] = 0.95  # Maximum fight confidence
        for i in range(3):
            result = self.engine.process(detection)
        
        self.assertEqual(result['state'], State.CRITICAL)
        self.assertGreaterEqual(result['threat_score'], 3.2)
        self.assertIn('SAVE_EVIDENCE', result['action'])
        self.assertIn('LOCAL_ALARM', result['action'])
        self.assertIn('NOTIFY_OPERATOR', result['action'])
        self.assertIn('DISPATCH_UAV', result['action'])
        print(f"✓ CRITICAL state (fight): score={result['threat_score']}, action={result['action']}")
    
    def test_multiple_persons_tracking(self):
        """Test engine with multiple persons being tracked"""
        # Person 1: Normal
        detection1 = self.create_detection(person_id=1)
        
        # Person 2: With knife - need more frames to reach ARMED
        detection2 = self.create_detection(person_id=2, knife_conf=0.7)
        
        # Process both persons multiple times
        for i in range(12):
            result1 = self.engine.process(detection1)
            result2 = self.engine.process(detection2)
        
        self.assertEqual(result1['state'], State.NORMAL)
        self.assertEqual(result2['state'], State.ARMED)
        self.assertNotEqual(result1['id'], result2['id'])
        print(f"✓ Multiple tracking: Person1={result1['state']}, Person2={result2['state']}")
    
    def test_state_transitions(self):
        """Test state transitions from normal through all levels"""
        person_id = 1
        
        # Start with normal
        detection = self.create_detection(person_id=person_id)
        result = self.engine.process(detection)
        self.assertEqual(result['state'], State.NORMAL)
        
        # Escalate to suspicious (loitering)
        detection['meta'] = {"loitering": True}
        for i in range(12):
            result = self.engine.process(detection)
        self.assertEqual(result['state'], State.SUSPICIOUS)
        
        # Escalate to armed (knife)
        detection['meta'] = {}
        detection['knife_conf'] = 0.6
        for i in range(5):
            result = self.engine.process(detection)
        self.assertEqual(result['state'], State.ARMED)
        
        # Escalate to violent (gun)
        detection['knife_conf'] = 0.0
        detection['gun_conf'] = 0.8
        for i in range(5):
            result = self.engine.process(detection)
        self.assertEqual(result['state'], State.VIOLENT)
        
        # Escalate to critical (fight)
        detection['gun_conf'] = 0.0
        detection['fight_conf'] = 0.9
        result = self.engine.process(detection)
        self.assertEqual(result['state'], State.CRITICAL)
        
        print(f"✓ State transitions: NORMAL -> SUSPICIOUS -> ARMED -> VIOLENT -> CRITICAL")
    
    def test_confidence_smoothing(self):
        """Test confidence smoothing with EMA and Kalman"""
        detection = self.create_detection(person_conf=0.9)
        
        # Process multiple frames to see smoothing effect
        scores = []
        for i in range(10):
            # Vary confidence slightly
            detection['person_conf'] = 0.9 + (i % 3) * 0.05
            result = self.engine.process(detection)
            scores.append(result['confidence'])
        
        # Check that smoothing is working (scores should be relatively stable)
        score_variance = np.var(scores)
        self.assertLess(score_variance, 0.01)  # Should be low variance due to smoothing
        print(f"✓ Confidence smoothing: variance={score_variance:.4f}")
    
    def test_threat_scoring_components(self):
        """Test all components of threat scoring"""
        detection = self.create_detection(gun_conf=0.8, knife_conf=0.6, fight_conf=0.3)
        result = self.engine.process(detection)
        
        # Check that all scoring components are present
        self.assertIn('severity', result)
        self.assertIn('bayes_prob', result)
        self.assertIn('confidence', result)
        self.assertIn('duration_frames', result)
        self.assertIn('threat_score', result)
        
        # Check ranges
        self.assertGreaterEqual(result['severity'], 0)
        self.assertLessEqual(result['bayes_prob'], 1.0)
        self.assertLessEqual(result['confidence'], 1.0)
        self.assertGreaterEqual(result['threat_score'], 0)
        
        print(f"✓ Threat scoring components: severity={result['severity']}, bayes={result['bayes_prob']:.3f}, confidence={result['confidence']:.3f}")
    
    def test_engine_configuration(self):
        """Test engine with custom configuration"""
        custom_config = {
            "gun_threshold": 0.3,  # Lower threshold
            "knife_threshold": 0.3,
            "critical_score": 2.8,  # Lower critical threshold
            "ema_alpha": 0.8  # Higher smoothing
        }
        
        custom_engine = DecisionEngine(custom_config)
        detection = self.create_detection(gun_conf=0.4)  # Should trigger with lower threshold
        
        result = custom_engine.process(detection)
        # Should escalate faster due to lower thresholds
        self.assertGreaterEqual(result['threat_score'], 2.0)
        
        print(f"✓ Custom configuration: score={result['threat_score']} with lowered thresholds")

class TestEnginePerformance(unittest.TestCase):
    """Performance and stress tests"""
    
    def test_high_volume_processing(self):
        """Test engine performance with high volume of detections"""
        engine = DecisionEngine()
        
        start_time = time.time()
        
        # Simulate 1000 detections across 10 persons
        for i in range(1000):
            person_id = i % 10
            detection = {
                "id": person_id,
                "bbox": [100, 100, 80, 200],
                "person_conf": 0.9,
                "gun_conf": 0.1 if person_id < 3 else 0.0,
                "knife_conf": 0.0,
                "fight_conf": 0.0,
                "timestamp": time.time()
            }
            engine.process(detection)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 1000 detections in reasonable time
        self.assertLess(processing_time, 2.0)  # Less than 2 seconds
        print(f"✓ Performance: 1000 detections processed in {processing_time:.3f}s")

def run_comprehensive_test():
    """Run all tests and display results"""
    print("=" * 60)
    print("HYBRID DECISION ENGINE - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTest(unittest.makeSuite(TestHybridDecisionEngine))
    suite.addTest(unittest.makeSuite(TestEnginePerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("🎉 ALL TESTS PASSED! Engine is ready for production.")
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
    
    return result

if __name__ == "__main__":
    run_comprehensive_test()
