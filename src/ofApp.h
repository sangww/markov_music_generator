#pragma once

#include "ofMain.h"
#include "ofxMidiFile.h"
#include "ofxMidi.h"
#include "ofxMSATensorFlow.h"

using namespace tensorflow;

class ofApp : public ofBaseApp{
	public:
		void setup();
		void update();
		void draw();
		
		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y);
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
    
    //midi read
    ofxMidiFile midi;
    int mt; //main track;
    ofxMidiFile gen;
    vector<vector<MidiFileEvent>> events, gen_events;
    
    //UI
    ofRectangle slider[2];
    int index_slider = -1;
    int offset_x, offset_y;
    bool bPlayGenerated = false;
    
    
    //ngram struct for hybrid
    struct hybrid {
        int pitch;
        int beat;
        
        bool operator==(const hybrid& rhs) const
        {
            if (pitch != rhs.pitch)
                return false;
            if (beat != rhs.beat)
                return false;
            return true;
        }
        bool operator!=(const hybrid& rhs) const
        {
            if (pitch != rhs.pitch)
                return true;
            if (beat != rhs.beat)
                return true;
            return false;
        }
        bool operator<(const hybrid &rhs) const
        {
            if(pitch < rhs.pitch)
                return true;
            else if (pitch > rhs.pitch)
                return false;
            else
                return beat < rhs.beat;
        }
        bool operator>(const hybrid &rhs) const
        {
            if(pitch > rhs.pitch)
                return true;
            else if (pitch < rhs.pitch)
                return false;
            else
                return beat > rhs.beat;
        }
    };
    struct compare_hybrid_b_rhythm {
        bool operator()(hybrid const &a, hybrid const &b) {
            return a.beat < b.beat;
        }
    };
    struct compare_hybrid_b_pitch {
        bool operator()(hybrid const &a, hybrid const &b) {
            return a.pitch < b.pitch;
        }
    };
    
    //hybrid ngram
    struct hybrid_ngram {
        vector<hybrid> gram;
        vector<hybrid> o_r_sorted, o_p_sorted;
        int cut_pitch = 0;
        int cut_rhythm[3] = {0, 0, 0};
    };
    struct compare_hybrid {
        bool operator()(hybrid_ngram const &a, hybrid_ngram const &b) {
            if(a.gram.size() < b.gram.size()) return true;
            else if (a.gram.size() > b.gram.size()) return false;
            else{
                return a.gram < b.gram;
            }
        }
    };


    //pitch and rhythm ngrams
    struct pitch_ngram {
        vector<int> gram;
        vector<int> o;
        int cut = 0;
    };
    struct rhythm_ngram {
        vector<int> gram;
        vector<int> o;
        int cut[3] = {0, 0, 0}; //8th, 16th, 32th
    };
    struct compare_pitch {
        bool operator()(pitch_ngram const &a, pitch_ngram const &b) {
            if(a.gram.size() < b.gram.size()) return true;
            else if (a.gram.size() > b.gram.size()) return false;
            else{
                return a.gram < b.gram;
            }
        }
    };
    struct compare_rhythm {
        bool operator()(rhythm_ngram const &a, rhythm_ngram const &b) {
            if(a.gram.size() < b.gram.size()) return true;
            else if (a.gram.size() > b.gram.size()) return false;
            else{
                return a.gram < b.gram;
            }
        }
    };
    
    
    
    //Markov Model
    int multiplier;
    int beat_division;
    int size_n;
    int cpb;
    int tempo;
    
    //composition
    void generateMidi(int num, string fn, int fNote = 64, int fBeat = 0, int tempo = 120, int clickspb = 480);
    void saveMidi(string fn);
    bool isGenerated;
    int firstNote, firstBeat;
    
    //ngram
    vector<pitch_ngram> ngram_p;
    vector<rhythm_ngram> ngram_b;
    vector<hybrid_ngram> ngram_h;
    
    //utility functions
    int compareHybrid(int n, vector<hybrid> p1, vector<hybrid_ngram> p2);
    int compareNote(int n, vector<int> p1, vector<pitch_ngram> p2);
    int compareRhythm(int n, vector<int> p1, vector<rhythm_ngram> p2);
    
    hybrid pollHybrid_b_pitch(int index, vector<hybrid_ngram> p, int option = -1);
    hybrid pollHybrid_b_rhythm(int index, vector<hybrid_ngram> p, int option = -1);
    
    int pollNote(int index, vector<pitch_ngram> p, int option = -1);
    int pollRhythm(int index, vector<rhythm_ngram> p, int option = -1);
    
    vector<hybrid> extractHybridSubVector(int n, vector<hybrid> p);
    vector<int> extractNoteOrBeatSubVector(int n, vector<int> p);
    
    void printHybrid(vector<hybrid_ngram> p, bool p_sorted = true);
    void printNote(vector<pitch_ngram> p);
    void printRhythm(vector<rhythm_ngram> p);
    void printNgramWithoutFormat(int n, vector<hybrid> p);
    
    void saveNgramToFile(string fn, vector<hybrid_ngram> p);
    
    //interface with external midi systems    
    ofxMidiOut midiOut;
    void midiEventCallback(MidiFileEvent & e);
    
    
    //tensorflow
    msa::tf::Session_ptr session;
    msa::tf::GraphDef_ptr graph_def;
    Tensor a, b, accuracy;
    vector<float> a_vec, b_vec, accuracy_vec;
    vector<Tensor> outputs;
    
    void generateMidiNeuralNet(int num, string fn, int fNote = 64, int fBeat = 0, int tempo = 120, int clickspb = 480);
};
