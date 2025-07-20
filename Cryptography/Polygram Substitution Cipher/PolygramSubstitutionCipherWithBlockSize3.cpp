#include<bits/stdc++.h>

using namespace std;

map<string, string> encryptMap, decryptMap;
vector<string>subst[3];

void createSubs() {
    // for block size 3.
    for(char ch = 'A'; ch <= 'Z'; ch++) {
        for(char ch1 = 'A'; ch1 <= 'Z'; ch1++) {
            for(char ch2 = 'A'; ch2 <= 'Z'; ch2++) {
                string tem = string(1,ch) + string(1,ch1) + string(1,ch2);
                subst[2].push_back(tem);
            }
        }
    }
    
    // for block size 2.
    for(char ch1 = 'A'; ch1 <= 'Z'; ch1++) {
        for(char ch2 = 'A'; ch2 <= 'Z'; ch2++) {
            string tem = string(1,ch1) + string(1,ch2);
            subst[1].push_back(tem);
        }
    }

    // for block size 1.
    for(char ch2 = 'A'; ch2 <= 'Z'; ch2++) {
        string tem = string(1,ch2);
        subst[0].push_back(tem);
    }

    random_device rd;
    mt19937 g(rd());

    for(int i=0;i<3;i++) {
        vector<string> tem = subst[i];
        shuffle(tem.begin(), tem.end(), g);
        
        for (int j=0;j<tem.size();i++) {
            encryptMap[subst[i][j]] = tem[j];
            decryptMap[tem[j]] = subst[i][j];
        }
    }


}

void polygramSubstitution(string str, int key) {
    
}

int main() {
    string str;
    cout << "Enter Your Message:" << endl;
    getline(cin, str);

    createSubs();

    int block_size = 3;

    return 0;

}