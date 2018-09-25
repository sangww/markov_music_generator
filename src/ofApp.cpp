#include "ofApp.h"
const unsigned int NUM_NOTES = 240;
const unsigned int MAX_GAP = 32;
const unsigned int MAX_EVENTS = 5000;
const bool debug = false;


void ofApp::generateMidi(int num, string fn, int fNote, int fBeat, int tempo, int clickspb){ //weighted-sorted generation

    gen.clear();
    gen.setTrackNum(1);
    gen.setTempo(tempo);
    gen.setClicksPerBeat(clickspb);
    
    //values continuously updated
    int note = fNote;
    int gap = 1;
    
    //randomly fixed values
    int vel_on = 72, vel_off = 64;
    float dur = 1.0f * multiplier;
    
    //track time
    int time = fBeat;
    float beat_inverse_division = 1 / (float) beat_division;
    
    vector<hybrid> ng_hyb;
    vector<int> ng_pitch;
    vector<int> ng_beat;
    
    //first note
    gen.addNoteOn(note, vel_on, time * beat_inverse_division * cpb, 0);
    gen.addNoteOff(note, vel_off, (time + dur) * beat_inverse_division * cpb, 0);
    
    //add to tracked ngrams
    ng_pitch.push_back(note);
    ng_beat.push_back(0);
    hybrid pr; pr.pitch = note; pr.beat = 0;
    ng_hyb.push_back(pr);
    
    int idx;
    cout << "[composing...]" << endl;;
    
    for(int i = 0; i < num; i++){
        int j;
        bool bNote = false;
        bool bBeat = false;
        
        for(j = size_n; j > 0; j--){
            if(ng_hyb.size() < j) continue;
            
            int idx = compareHybrid(j, ng_hyb, ngram_h);
            if(idx < 0){
                if(!bNote){ //note seperately
                    idx = compareNote(j, ng_pitch, ngram_p);
                    if(idx < 0){
                        note = fNote; //default just in case everything fails
                    }
                    else{ //at least temporarily have the best result
                        note = pollNote(idx, ngram_p);
                        bNote = true;
                    }
                }
                
                if(!bBeat){ //beat seperately
                    idx = compareRhythm(j, ng_beat, ngram_b);
                    if(idx < 0){
                        gap = beat_division - ng_beat[ng_beat.size() - 1]; //default (zeroing the next beat) in case all fail
                    }
                    else{
                        gap = pollRhythm(idx, ngram_b);
                        bBeat = true;
                    }
                }
                
                if(bNote && bBeat){ //we found a match
                    cout << j << "=";
                    break;
                }
                else if(j == 1){ //failed at everything: we use the defaulting options
                    if(bNote) cout << "o";
                    else cout << "x";
                    if(bBeat) cout << "o";
                    else cout << "x";
                }
                else{
                    continue; //we can try more options (either seperate or hybrid)
                }
            }
            else{
                pr = pollHybrid_b_pitch(idx, ngram_h);
                if(ng_hyb[ng_hyb.size()-1].pitch == pr.pitch && pr.beat == 0){ //remove duplicate
                    if(j > 1) continue;
                    else cout << "zz"; //failed at everything: we use the defaulting options
                }
                else{ //found a match
                    cout << j << "-";
                    note = pr.pitch;
                    gap = pr.beat;
                    break;
                }
            }
        }
        
        time = time + gap;
        pr.pitch = note; pr.beat = time % beat_division;
        
        gen.addNoteOn(note, vel_on, time * beat_inverse_division * cpb, 0);
        gen.addNoteOff(note, vel_off, (time + dur) * beat_inverse_division * cpb, 0);
        
        if(debug){
            cout << i<< ": ";
            printNgramWithoutFormat(j, ng_hyb);
            cout << " "<<note << " " <<gap <<" " << time <<endl;
        }
        
        //add to tracking
        ng_pitch.push_back(note);
        if(ng_pitch.size() > size_n) ng_pitch.erase(ng_pitch.begin());
        
        ng_beat.push_back(time % beat_division);
        if(ng_beat.size() > size_n) ng_beat.erase(ng_beat.begin());
        
        ng_hyb.push_back(pr);
        if(ng_hyb.size() > size_n) ng_hyb.erase(ng_hyb.begin());
    }
    cout << endl;

    gen.golive();
    while(!gen.isParsed());
    gen_events.clear();
    gen_events = gen.getEvents();
    isGenerated = true;
}

void ofApp::generateMidiNeuralNet(int num, string fn, int fNote, int fBeat, int tempo, int clickspb){ //weighted-sorted generation
    
    gen.clear();
    gen.setTrackNum(1);
    gen.setTempo(tempo);
    gen.setClicksPerBeat(clickspb);
    
    //values continuously updated
    int note = fNote;
    int gap = 1;
    
    //randomly fixed values
    int vel_on = 72, vel_off = 64;
    float dur = 1.0f * multiplier;
    
    //track time
    int time = fBeat;
    float beat_inverse_division = 1 / (float) beat_division;
    
    vector<hybrid> ng_hyb;
    vector<int> ng_pitch;
    vector<int> ng_beat;
    
    //first note
    gen.addNoteOn(note, vel_on, time * beat_inverse_division * cpb, 0);
    gen.addNoteOff(note, vel_off, (time + dur) * beat_inverse_division * cpb, 0);
    
    //add to tracked ngrams
    ng_pitch.push_back(note);
    ng_beat.push_back(0);
    hybrid pr; pr.pitch = note; pr.beat = 0;
    ng_hyb.push_back(pr);
    
    cout << "[composing using tensorflow...]" << endl;
    
    if(session) {
        for(int i = 0; i < num; i++){
            //assign to tf input tensors
            for(int j = 0; j < a_vec.size() - 1; j --){
                if( j == 0){
                    a_vec[a_vec.size() - 1] = ng_hyb[ng_hyb.size() - 1].beat/128.f;
                }
                else if( j < ng_hyb.size() + 1){
                    a_vec[a_vec.size() - 1 - j] = ng_hyb[ng_hyb.size() - j].pitch/128.f;
                }
                else{
                    a_vec[a_vec.size() - 1 - j] = 0;
                }
            }
            msa::tf::vector_to_tensor(a_vec, a);
            
            // IMPORTANT: the string must match the name of the variable/node in the graph
            vector<pair<string, Tensor>> inputs = {
                { "x", a }
            };
            
            // desired outputs which we want processed and returned from the graph
            // IMPORTANT: the string must match the name of the variable/node in the graph
            vector<string> output_names = { "y_out" };
            
            // Run the graph, pass in our inputs and desired outputs, evaluate operation and return
            if(!session->Run(inputs, output_names, {}, &outputs).ok()) {
                ofLogError() << "Error during running.";
                return false;
            }
            
            // outputs is a vector of tensors, we're interested in only the first tensor
            msa::tf::tensor_to_vector(outputs[0], b_vec);
            
            //read nn output
            note = (int)(b_vec[0] * 128.f + 0.5f);
            gap = (int)(b_vec[1] * 128.f + 0.5f);
            
            time = time + gap;
            pr.pitch = note; pr.beat = time % beat_division;
            
            gen.addNoteOn(note, vel_on, time * beat_inverse_division * cpb, 0);
            gen.addNoteOff(note, vel_off, (time + dur) * beat_inverse_division * cpb, 0);
            
            //if(debug){
                cout << i<< ": "<<note << " " <<gap <<" " << time <<endl;
            //}
            
            //add to tracking
            ng_pitch.push_back(note);
            if(ng_pitch.size() > size_n) ng_pitch.erase(ng_pitch.begin());
            
            ng_beat.push_back(time % beat_division);
            if(ng_beat.size() > size_n) ng_beat.erase(ng_beat.begin());
            
            ng_hyb.push_back(pr);
            if(ng_hyb.size() > size_n) ng_hyb.erase(ng_hyb.begin());
        }
        
        gen.golive();
        while(!gen.isParsed());
        gen_events.clear();
        gen_events = gen.getEvents();
        isGenerated = true;
        
    } else {
        cout << "Error during tensorflow initialization." <<endl;
    }
}

void ofApp::saveMidi(string fn){
    if(isGenerated) gen.save(fn);
}

//--------------------------------------------------------------
void ofApp::setup(){
    ofBackground(0);
    ofSetWindowShape(800, 400);
    //ofSetLogLevel(OF_LOG_VERBOSE);
    
    //set midi out
    ofAddListener(MidiFileEvent::NOTE_ON, this, &ofApp::midiEventCallback);
    ofAddListener(MidiFileEvent::NOTE_OFF, this, &ofApp::midiEventCallback);
    midiOut.listPorts(); // via instance
    //midiOut.openPort(0); // by number
    //midiOut.openPort("IAC Driver Pure Data In"); // by name
    midiOut.openVirtualPort("ofxMidiOut"); // open a virtual port
    
    //load midi
    midi.load("under.mid");
    mt = 1; //to be tweaked once the midi has different setting
    
    while(!midi.isParsed());
    if(mt > midi.getTrackNum() - 1) mt = midi.getTrackNum() -1;
    
    //read some variables needed
    events = midi.getEvents();
    cpb = midi.getClicksPerBeat();
    tempo = midi.getTempo();
    
    //midi parameters extra - user defined
    multiplier = 4;
    beat_division = 4 * multiplier;
    size_n = 4;
    
    //load individual parameters
    float t_cur = 0.f;
    vector<hybrid> ng_hyb;
    vector<int> ng_pitch;
    vector<int> ng_beat;
    
    //analysis
    int cnt = 0;
    int progress = 0;
    int num = events[mt].size();
    if(num > MAX_EVENTS) num = MAX_EVENTS;
    
    for(int i = 0; i < num; i++){
        if(100 * i / num - progress >= 10){
            progress = 100*i/num;
            cout << "-";
            if(i == num -1) cout <<endl;
        }
        
        //inspect first N items
        if(events[mt][i].note > 0 && cnt < 2){
            cout << "[start frames] " << events[mt][i].note << " " << events[mt][i].time / (float)midi.getClicksPerBeat()
                << " " << events[mt][i].type << " " << events[mt][i].velocity << endl;
            if(cnt == 0){
                firstNote = events[mt][i].note;
                firstBeat = (int) cpb * events[mt][i].time / (float)midi.getClicksPerBeat();
            }
            cnt ++;
        }
        
        if(events[mt][i].type == 0 && events[mt][i].note > 0){
            //read midi data
            float t_in = events[mt][i].time / (float)midi.getClicksPerBeat();
            int note = events[mt][i].note;
            int gap = beat_division * (t_in - t_cur);
            int beat = beat_division * (t_in - (int)t_in);
            hybrid pr; pr.pitch = note; pr.beat = gap;
            t_cur = t_in;
            
            //printf("[%d  %02d  %02d %f]\n", note, beat, gap, beat_division*t_in);
            
            if(ng_hyb.size() > 0 && ng_hyb[0].pitch > 0){
                //note
                for(int j = size_n; j > 0; j--){
                    if(ng_pitch.size() < j) continue;
                    
                    int idx = compareNote(j, ng_pitch, ngram_p);
                    if(idx < 0){
                        pitch_ngram s;
                        s.gram = extractNoteOrBeatSubVector(j, ng_pitch);
                        s.o.push_back(note);
                        ngram_p.push_back(s);
                    }
                    else{
                        ngram_p[idx].o.push_back(note);
                    }
                }
                
                //beat
                for(int j = size_n; j > 0; j--){
                    if(ng_beat.size() < j) continue;
                    
                    int idx = compareRhythm(j, ng_beat, ngram_b);
                    if(idx < 0){
                        rhythm_ngram s;
                        s.gram = extractNoteOrBeatSubVector(j, ng_beat);
                        s.o.push_back(gap);
                        if(gap /multiplier < MAX_GAP) ngram_b.push_back(s);
                    }
                    else{
                        if(gap /multiplier < MAX_GAP) ngram_b[idx].o.push_back(gap);
                    }
                }
                
                //hybrid
                for(int j = size_n; j > 0; j--){
                    if(ng_hyb.size() < j) continue;
                    
                    int idx = compareHybrid(j, ng_hyb, ngram_h);
                    if(idx < 0){
                        hybrid_ngram s;
                        s.gram = extractHybridSubVector(j, ng_hyb);
                        s.o_r_sorted.push_back(pr);
                        s.o_p_sorted.push_back(pr);
                        if(gap /multiplier < MAX_GAP) ngram_h.push_back(s);
                    }
                    else{
                        if(gap /multiplier < MAX_GAP) ngram_h[idx].o_r_sorted.push_back(pr);
                        if(gap /multiplier < MAX_GAP) ngram_h[idx].o_p_sorted.push_back(pr);
                    }
                }
            }
            
            ng_pitch.push_back(note);
            if(ng_pitch.size() > size_n) ng_pitch.erase(ng_pitch.begin());
            
            ng_beat.push_back(beat);
            if(ng_beat.size() > size_n) ng_beat.erase(ng_beat.begin());
            
            pr.beat = beat; //CIRITCAL ERROR
            ng_hyb.push_back(pr);
            if(ng_hyb.size() > size_n) ng_hyb.erase(ng_hyb.begin());
        }
    }
    cout << "[collected ngrams: " << ngram_p.size() << " " << ngram_b.size() << " " << ngram_h.size() << "]" << endl << endl;
    
    //sorting
    sort(ngram_h.begin(), ngram_h.end(), compare_hybrid());
    for(int i = 0; i < ngram_h.size(); i++){
        sort(ngram_h[i].o_p_sorted.begin(), ngram_h[i].o_p_sorted.end(), compare_hybrid_b_pitch());
        sort(ngram_h[i].o_r_sorted.begin(), ngram_h[i].o_r_sorted.end(), compare_hybrid_b_rhythm());
        
        for(int j = 0; j < ngram_h[i].o_p_sorted.size(); j++){
            if(ngram_h[i].o_p_sorted[j] < ngram_h[i].gram[ngram_h[i].gram.size() -1]){
                ngram_h[i].cut_pitch = j + 1;
            }
        }
        
        for(int j = 0; j < ngram_h[i].o_r_sorted.size(); j++){
            if(ngram_h[i].o_r_sorted[j].beat / multiplier < 1){
                ngram_h[i].cut_rhythm[0] = j + 1;
            }
            if(ngram_h[i].o_r_sorted[j].beat / multiplier < 2){
                ngram_h[i].cut_rhythm[1] = j + 1;
            }
            if(ngram_h[i].o_r_sorted[j].beat / multiplier < 4){
                ngram_h[i].cut_rhythm[2] = j + 1;
            }
        }
    }
    
    sort(ngram_p.begin(), ngram_p.end(), compare_pitch());
    for(int i = 0; i < ngram_p.size(); i++){
        sort(ngram_p[i].o.begin(), ngram_p[i].o.end());
        
        for(int j = 0; j < ngram_p[i].o.size(); j++){
            if(ngram_p[i].o[j] < ngram_p[i].gram[ngram_p[i].gram.size() -1]){
                ngram_p[i].cut = j + 1;
            }
        }
    }
    
    sort(ngram_b.begin(), ngram_b.end(), compare_rhythm());
    for(int i = 0; i < ngram_b.size(); i++){
        sort(ngram_b[i].o.begin(), ngram_b[i].o.end());
        
        for(int j = 0; j < ngram_b[i].o.size(); j++){
            if(ngram_b[i].o[j] / multiplier < 1){
                ngram_b[i].cut[0] = j + 1;
            }
            if(ngram_b[i].o[j] / multiplier < 2){
                ngram_b[i].cut[1] = j + 1;
            }
            if(ngram_b[i].o[j] / multiplier < 4){
                ngram_b[i].cut[2] = j + 1;
            }
        }
    }
    isGenerated = false;
    
    //tensorflow
    // Load graph (i.e. trained model) we exported from python, add to session, return if error
    graph_def = msa::tf::load_graph_def("models/graph-mlp.pb");
    if(!graph_def) return;
    
    // initialize session with graph
    session = msa::tf::create_session_with_graph(graph_def);
    
    a_vec.reserve(size_n + 1);
    for (int i = 0; i < size_n + 1; i++) a_vec.push_back(0);
    b_vec.reserve(2);
    b_vec.push_back(0); b_vec.push_back(0);
    accuracy_vec.reserve(0);
    accuracy_vec.push_back(0);
    a = Tensor(DT_FLOAT, TensorShape( {1, size_n + 1}));
    b = Tensor(DT_FLOAT, TensorShape( {1, 2}));
    accuracy =Tensor(DT_FLOAT, TensorShape( {1, 1}));
    
    //for hacking numbers into existing nodes (currently not needed)
    std::vector<string> names;
    int node_count = graph_def->node_size();
    ofLogNotice() << "Node Count - " << node_count;
    
    for(int i=0; i<node_count; i++) {
        auto n = graph_def->node(i);
        if(n.name().find("_ASSIGNCONST") != std::string::npos) {
            ofLogNotice() << i << ":" << n.name();
            names.push_back(n.name());
        }
        if(n.name().find("_ASSIGNVAR") != std::string::npos) {
            ofLogNotice() << i << ":" << n.name();
        }
        if(n.name().find("_SAVEDCONST") != std::string::npos) {
            ofLogNotice() << i << ":" << n.name();
        }
        if(n.name().find("_SAVEDVAR") != std::string::npos) {
            ofLogNotice() << i << ":" << n.name();
        }
    }
    
    if(!session->Run({}, names, {}, &outputs).ok()) {
        ofLogError() << "Error running network for weights and biases variable hack";
        return false;
    }

    //UI
    slider[0].setSize(20, 20);
    slider[1].setSize(20, 20);
}

//--------------------------------------------------------------
void ofApp::update(){
    
}

//--------------------------------------------------------------
void ofApp::draw(){
    ofSetColor(150, 240, 255);
    ofPushMatrix();
    ofTranslate(20, 20);
    ofDrawBitmapString(ofToString(ofGetFrameRate()) + " fps", 0, 0);
    ofDrawBitmapString(ofToString(tempo) + " bpm", 0, 20);
    ofPopMatrix();
    ofSetLineWidth(0.5);
    
    if(!bPlayGenerated) ofSetColor(200);
    else ofSetColor(100);
    ofDrawLine(20, 80, 420, 80);
    if(midi.isLoaded()) ofSetColor(200);
    if(midi.isPlaying()) ofSetColor(150, 240, 255);
    slider[0].setPosition(midi.getPositionPercent()*400 + 10, 70);
    ofDrawRectangle(slider[0].getX(), slider[0].getY(), slider[0].getWidth(), slider[0].getHeight());
    
    
    if(bPlayGenerated) ofSetColor(200);
    else ofSetColor(100);
    ofDrawLine(20, 240, 420, 240);
    if(isGenerated){
        ofSetColor(200);
        if(gen.isPlaying()) ofSetColor(150, 240, 255);
        slider[1].setPosition(gen.getPositionPercent()*400 + 10, 230);
    }
    else{
        slider[1].setPosition(10, 230);
    }
    ofDrawRectangle(slider[1].getX(), slider[1].getY(), slider[1].getWidth(), slider[1].getHeight());
    ofPopMatrix();
    
    //draw midi generation
    if(midi.isLoaded()){
        ofPushMatrix();
        ofTranslate(20, 200);
        float inverseCpb = 1 / (float)midi.getClicksPerBeat();
        for(int i = 0; i < events[mt].size(); i++){
            int x = events[mt][i].time * beat_division * inverseCpb * 0.5;
            int y = - events[mt][i].note;
            ofSetColor(150);
            if( fabs( events[mt][i].time) - midi.msToMidiClockTicks(midi.getPositionMS()) < 2){
                ofSetColor(255, 0, 0);
            }
            ofDrawRectangle(x, y, 1, 1);
        }
        ofPopMatrix();
    }
    if(isGenerated){
        if(midi.isLoaded()){
            ofPushMatrix();
            ofTranslate(20, 350);
            float inverseCpb = 1 / (float)midi.getClicksPerBeat();
            for(int i = 0; i < gen_events[0].size(); i++){
                int x = gen_events[0][i].time * beat_division * inverseCpb * 0.5;
                int y = - gen_events[0][i].note;
                ofSetColor(150);
                if( fabs( gen_events[0][i].time) - gen.msToMidiClockTicks(gen.getPositionMS()) < 2){
                    ofSetColor(255, 0, 0);
                }
                ofDrawRectangle(x, y, 1, 1);
            }
            ofPopMatrix();
        }
    }
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    if(key == 'g'){
        generateMidi(NUM_NOTES, "gen-b" + ofToString(beat_division) + "w" + ofToString(size_n) + "-" + ofGetTimestampString() + ".midi", firstNote, 0, tempo, cpb);
    }
    else if(key =='1'){
        printNote(ngram_p);
    }
    else if(key =='2'){
        printRhythm(ngram_b);
    }
    else if(key =='3'){
        printHybrid(ngram_h, true);
    }
    else if(key =='4'){
        printHybrid(ngram_h, false);
    }
    else if(key ==' '){
        if(bPlayGenerated){
            if(gen.isPlaying()){
                gen.stop();
            }
            else{
                gen.play();
            }
            if(midi.isPlaying()) midi.stop();
        }
        else{
            if(midi.isPlaying()){
                midi.stop();
            }
            else{
                midi.play();
            }
            if(gen.isPlaying()) gen.stop();
        }
    }
    else if(key == 'i'){
        midi.stop();
        midi.setPositionPercent(0.f);
        gen.stop();
        gen.setPositionPercent(0.f);
    }
    else if(key == '[' || key == ']'){
        if(isGenerated) bPlayGenerated = !bPlayGenerated;
    }
    else if(key == 's'){
        saveMidi("gen-b" + ofToString(beat_division) + "w" + ofToString(size_n) + "-" + ofGetTimestampString() + ".midi");
    }
    else if(key =='x'){
        midiOut << StartMidi() << 0x01 << 0x7B << 0x00 << FinishMidi();
    }
    else if(key == 'v'){
        saveNgramToFile("nn.mkm", ngram_h);
    }
    else if(key == 'n'){
        generateMidiNeuralNet(NUM_NOTES, "gen-b" + ofToString(beat_division) + "w" + ofToString(size_n) + "-" + ofGetTimestampString() + ".midi", firstNote, 0, tempo, cpb);
    }
    else if(key == '-'){
        tempo -= 10;
        if(tempo < 50) tempo = 50;
        gen.setTempo(tempo);
    }
    else if(key == '='){
        tempo += 10;
        if(tempo > 500) tempo = 500;
        gen.setTempo(tempo);
    }
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button) {
    if(index_slider >= 0){
        if(index_slider == 0){
            if(midi.isLoaded() && midi.isPlaying()){
                midi.stop();
                midi.setPositionPercent(ofMap(x - 20, 0, 400, 0, 1.f, true));
                midi.play();
            }
            if(midi.isLoaded()){
                midi.setPositionPercent(ofMap(x - 20, 0, 400, 0, 1.f, true));
            }
        }
        else if(index_slider == 1){
            if(isGenerated && gen.isPlaying()){
                gen.stop();
                gen.setPositionPercent(ofMap(x - 20, 0, 400, 0, 1.f, true));
                gen.play();
            }
            if(isGenerated){
                gen.setPositionPercent(ofMap(x - 20, 0, 400, 0, 1.f, true));
            }
        }
    }
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button) {
    index_slider = -1;
    for(int i = 0; i < 2; i++){
        if(slider[i].inside(x, y)){
            index_slider = i;
            offset_x = x - slider[i].getX();
            offset_y = y - slider[i].getY();
        }
    }
    if(index_slider < 0){
        if(index_slider == 0){
            if(x > 20 && x < 420 && y > 70 && y < 90){
                if(midi.isLoaded() && midi.isPlaying()){
                    midi.stop();
                    midi.setPositionPercent(ofMap(x - 20, 0, 400, 0, 1.f, true));
                    midi.play();
                }
                if(midi.isLoaded()){
                    midi.setPositionPercent(ofMap(x - 20, 0, 400, 0, 1.f, true));
                }
            }
        }
        else
            if(index_slider == 1){
                if(x > 20 && x < 420 && y > 230 && y < 250){
                    if(isGenerated && gen.isPlaying()){
                        gen.stop();
                        gen.setPositionPercent(ofMap(x - 20, 0, 400, 0, 1.f, true));
                        gen.play();
                    }
                    if(isGenerated){
                        gen.setPositionPercent(ofMap(x - 20, 0, 400, 0, 1.f, true));
                    }
                }
            }
    }
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button ) {
    index_slider = -1;
}


//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}


//// compare and get indox of an ngram
int ofApp::compareHybrid(int n, vector<hybrid> p1, vector<hybrid_ngram> p2){
    int window = min(n, size_n);
    
    for(int j = 0; j < p2.size(); j++){
        if(p2[j].gram.size() == window){
            bool b = true;
            for(int k = p2[j].gram.size() - window; k < p2[j].gram.size() - 1; k ++){
                if(p1[k].pitch != p2[j].gram[k].pitch){
                    b = false;
                    break;
                }
            }
            if(b && p1[p1.size() -1] == p2[j].gram[p2[j].gram.size() - 1]){
                return j;
            }
        }
    }
    return -1;
}

int ofApp::compareNote(int n, vector<int> p1, vector<pitch_ngram> p2){
    int window = min(n, size_n);
    
    for(int j = 0; j < p2.size(); j++){
        if(p2[j].gram.size() == window){
            bool b = true;
            for(int k = p2[j].gram.size() - window; k < p2[j].gram.size(); k ++){
                if(p1[k] != p2[j].gram[k]){
                    b = false;
                    break;
                }
            }
            if(b) return j;
        }
    }
    return -1;
}

int ofApp::compareRhythm(int n, vector<int> p1, vector<rhythm_ngram> p2){
    int window = min(n, size_n);
    
    for(int j = 0; j < p2.size(); j++){
        if(p2[j].gram.size() == window){
            bool b = true;
            for(int k = p2[j].gram.size() - window; k < p2[j].gram.size(); k ++){
                if(p1[k] != p2[j].gram[k]){
                    b = false;
                    break;
                }
            }
            if(b) return j;
        }
    }
    return -1;
}

//poll an output walk out of the matrix. TODO: optional sorted polling
ofApp::hybrid ofApp::pollHybrid_b_pitch(int index, vector<hybrid_ngram> p, int option){
    int rand = (int) ofRandom(0, p[index].o_p_sorted.size());
    int note = p[index].o_p_sorted[rand].pitch;
    int gap = p[index].o_p_sorted[rand].beat;
    hybrid pr; pr.pitch = note; pr.beat = gap;
    return pr;
}

ofApp::hybrid ofApp::pollHybrid_b_rhythm(int index, vector<hybrid_ngram> p, int option){
    int rand = (int) ofRandom(0, p[index].o_r_sorted.size());
    int note = p[index].o_r_sorted[rand].pitch;
    int gap = p[index].o_r_sorted[rand].beat;
    hybrid pr; pr.pitch = note; pr.beat = gap;
    return pr;
}

int ofApp::pollNote(int index, vector<pitch_ngram> p, int option){
    int rand = (int) ofRandom(0, p[index].o.size());
    return p[index].o[rand];
}

int ofApp::pollRhythm(int index, vector<rhythm_ngram> p, int option){
    int rand = (int) ofRandom(0, p[index].o.size());
    return p[index].o[rand];
}


//segmenting ngram
vector<ofApp::hybrid> ofApp::extractHybridSubVector(int n, vector<hybrid> p){
    vector<hybrid> v;
    if(p.size() >= n){
        for(int i = p.size() - n; i < p.size(); i++){
            v.push_back(p[i]);
        }
        return v;
    }
    else return;
}

vector<int> ofApp::extractNoteOrBeatSubVector(int n, vector<int> p){
    vector<int> v;
    if(p.size() >= n){
        for(int i = p.size() - n; i < p.size(); i++){
            v.push_back(p[i]);
        }
        return v;
    }
    else return;
}

//print markov analysis
void ofApp::printHybrid(vector<hybrid_ngram> p, bool p_sorted){
    cout << endl << "ngram hybrid ";
    if(p_sorted) cout << "sorted by pitch: " << endl;
    else cout << "sorted by rhythm: " << endl;
    
    for (int i = 0; i < p.size(); i ++){
        cout << "{" ;
        for (int j = 0; j < p[i].gram.size(); j++){
            if(j < p[i].gram.size() - 1){
                cout << p[i].gram[j].pitch << ",";
            }
            else{
                cout << "[" << p[i].gram[j].pitch <<"," << p[i].gram[j].beat << "]";
            }
        }
        if(p_sorted){
            cout <<"} - [" << p[i].cut_pitch << "] - ";
            for (int j = 0; j < p[i].o_p_sorted.size(); j++){
                cout << "(" << p[i].o_p_sorted[j].pitch <<"," <<p[i].o_p_sorted[j].beat << ")";
            }
        }
        else{
            cout <<"} - [" << p[i].cut_rhythm[0] << " " <<  p[i].cut_rhythm[1] << " " <<  p[i].cut_rhythm[2] << "] - ";
            for (int j = 0; j < p[i].o_r_sorted.size(); j++){
                cout << "(" << p[i].o_r_sorted[j].pitch <<"," <<p[i].o_r_sorted[j].beat << ")";
            }
        }
        cout << endl;
    }
}

void ofApp::printNote(vector<pitch_ngram> p){
    cout << endl << "ngram pitch: " << endl;
    for (int i = 0; i < p.size(); i ++){
        cout << "{" ;
        for (int j = 0; j < p[i].gram.size(); j++){
            cout << p[i].gram[j];
            if(j < p[i].gram.size() - 1) cout << ",";
        }
        cout <<"} - [" << p[i].cut << "] - ";
        for (int j = 0; j < p[i].o.size(); j++){
            cout << p[i].o[j] <<" ";
        }
        cout << endl;
    }
}

void ofApp::printRhythm(vector<rhythm_ngram> p){
    cout << endl << "ngram rhythm: " << endl;
    for (int i = 0; i < p.size(); i ++){
        cout << "{" ;
        for (int j = 0; j < p[i].gram.size(); j++){
            cout << p[i].gram[j];
            if(j < p[i].gram.size() - 1) cout << ",";
        }
        cout <<"} - [" << p[i].cut[0] << " " <<  p[i].cut[1] << " " <<  p[i].cut[2] << "] - ";
        for (int j = 0; j < p[i].o.size(); j++){
            cout << p[i].o[j] <<" ";
        }
        cout << endl;
    }
}

void ofApp::printNgramWithoutFormat(int n, vector<hybrid> p){
    cout << "{" ;
    for (int i = 0; i < p.size(); i ++){
        if( i == p.size() - n || (i==0 && n >= p.size())) cout <<"|";
        cout << p[i].pitch << "," << p[i].beat;
        if( i < p.size() -1 ) cout << " ";
        if( n == 0 && i == p.size() - 1) cout << "|";
    }
    cout << "}";
}

void ofApp::saveNgramToFile(string fn, vector<hybrid_ngram> p){
    cout << "[saving to file " << fn << "..." <<endl;
    ofFile file;
    file.open(fn,ofFile::WriteOnly);
    
    for (int i = 0; i < p.size(); i ++){
        for (int j = 0; j < p[i].gram.size(); j++){
            if(j < p[i].gram.size() - 1){
                file << p[i].gram[j].pitch << " ";
            }
            else{
                file << p[i].gram[j].pitch <<" " << p[i].gram[j].beat << ", ";
            }
        }
        for (int j = 0; j < p[i].o_p_sorted.size(); j++){
            file << p[i].o_p_sorted[j].pitch << " " <<p[i].o_p_sorted[j].beat << " ";
        }
        file << endl;
    }
    file.close();
    cout << "done." <<endl;
}



//midi event handler
void ofApp::midiEventCallback(MidiFileEvent & e){
    if(e.type == MIDIEVENT_TYPE::MIDIEVENT_NOTE_ON){
        midiOut.sendNoteOn(1, e.note, e.velocity);
    }
    if(e.type == MIDIEVENT_TYPE::MIDIEVENT_NOTE_OFF){
        midiOut.sendNoteOff(1, e.note, e.velocity);
    }
}
