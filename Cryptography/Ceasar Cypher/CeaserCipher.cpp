#include<bits/stdc++.h>

using namespace std;

string plainTextToCeaserCipher(string &str, int x) {

    string cipherText = "";
    for(char ch : str) {
        if (ch>='A' && ch<='Z') {
            int tem = ch - 'A';
            tem = (tem + x) % 26;
            ch = 'A' + tem;
        }
        else if(ch>='a' && ch<='z'){
            int tem = ch - 'a';
            tem = (tem + x) % 26;
            ch = 'a' + tem;
        }
        cipherText += ch;
    }

    cout << "        Plain Text: " << str << endl;
    cout << "Ceaser Cipher Text: " << cipherText << endl;
    cout << "\n\n";

    return cipherText;

}

string ceaserCipherToPlainText(string &str, int x) {

    string plainText = "";
    for(char ch : str) {
        if (ch>='A' && ch<='Z') {
            int tem = ch - 'A';
            tem = ((tem - x) + 26) % 26;
            ch = 'A' + tem;
        }
        else if(ch>='a' && ch<='z'){
            int tem = ch - 'a';
            tem = ((tem - x) + 26) % 26;
            ch = 'a' + tem;
        }
        plainText += ch;
    }

    cout << "Ceaser Cipher Text: " << str << endl;
    cout << "        Plain Text: " << plainText << endl;
    cout << "\n\n";
    

    return plainText;

}


int main() {
    string str;
    cout << "Enter Your Message:" << endl;
    getline(cin, str);

    int key;
    cout << "\nEnter your Key to Encrypt message (in range [1-25]):\n";
    cin >> key;

    cout << "\nEncryption:\n";
    string cipherText = plainTextToCeaserCipher(str, key);

    cout << "\nEnter your Key to Encrypt message (in range [1-25]):\n";
    cin >> key;

    cout << "\nDecryption:\n";
    string plainText = ceaserCipherToPlainText(cipherText, key);

    return 0;

}