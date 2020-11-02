#include <iostream>

using namespace std;

int comparator(const string &str1, const string &str2) {
    if (str1.length() != str2.length()) {
        return int(str1.length() - str2.length());
    }
    return str1.compare(str2);
}

void add_lid_zer_low(string &str1, const string &str2) {
    while (str1.length() < str2.length()) {
        str1.insert(0, "0");
    }
}

void remove_lid_zers(string &str) {
    while (str[0] == '0' && not str.empty()) {
        str.erase(0, 1);
    }
}

void add_lid_zers_both(string &str1, string &str2) {
    add_lid_zer_low(str1, str2);
    add_lid_zer_low(str2, str1);
}

void transform(string &str) {
    for (unsigned long i = str.length(); i > 1; --i) {
        if (str[i - 1] > '9') {
            str[i - 2] += (str[i - 1] - 0x30) / 10;
            str[i - 1] = char((str[i - 1] - 0x30) % 10 + 0x30);
        } else if (str[i - 1] < '0') {
            str[i - 2] -= (10 + 0x30 - str[i - 1]) / 10;
            str[i - 1] = char((10 - (0x30 - str[i - 1])) % 10 + 0x30);
        }
    }
    if (str[0] > '9') {
        str.insert(0, 1, char(((str[0] - 0x30) / 10) + 0x30));
        str[1] = char((str[1] - 0x30) % 10 + 0x30);
    }
}

string pls(string str1, string str2) {
    add_lid_zers_both(str1, str2);
    for (unsigned long i = str1.length(); i > 0; --i) {
        str1[i - 1] += str2[i - 1] - 0x30;
    }
    return str1;
}

string mns(string str1, string str2) {
    add_lid_zers_both(str1, str2);
    for (unsigned long i = str1.length(); i > 0; --i) {
        str1[i - 1] -= str2[i - 1] - 0x30;
    }
    transform(str1);
    return str1;
}

string multi(const string &str1, const string &str2) {
    string results[str1.length()];
    for (auto &i : results) {
        for (int j = 0; j < str2.length(); ++j) {
            i.append("0");
        }
    }
    for (unsigned long i = str1.length(); i > 0; --i) {
        for (unsigned long j = str2.length(); j > 0; --j) {
            results[i - 1][j - 1] = char(0x30 + (str1[i - 1] - 0x30) * (str2[j - 1] - 0x30));
        }
        transform(results[i - 1]);
        for (int j = 0; j < int(str1.length() - i); ++j) {
            results[i - 1].append("0");
        }
    }
    string ans;
    for (auto &i : results) {
        add_lid_zers_both(ans, i);
        ans = pls(ans, i);
    }
    transform(ans);
    return ans;
}

int low_div(string str1, const string &str2) {
    int res = 0;
    string current = str2;
    add_lid_zers_both(current, str1);
    while (comparator(str1, current) >= 0) {
        current = pls(current, str2);
        transform(current);
        ++res;
    }
    return res;
}

string lov_mod(const string &str1, const string &str2) {
    return mns(str1, multi(to_string(low_div(str1, str2)), str2));
}

string div(const string &str1, const string &str2) {

    string dividend;
    string ans;
    int i = 0;
    while (i < str1.length()) {
        if (dividend.empty() && str1[i] == '0') {
            ans.append("0");
            ++i;
            continue;
        }
        for (; comparator(dividend, str2) < 0 and i < str1.length(); ++i) {
            dividend.append(string(1, str1[i]));
            if (comparator(dividend, str2) < 0) {
                ans.append("0");
            }
        }
        if (i < str1.length() or low_div(dividend, str2) != 0) {
            ans.append(to_string(low_div(dividend, str2)));
        }
        dividend = lov_mod(dividend, str2);
        remove_lid_zers(dividend);
        if (dividend == "0") {
            dividend = "";
        }

    }
    remove_lid_zers(ans);
    if (ans.empty()) {
        ans.append("0");
    }
    return ans;
}

string mod(const string &str1, const string &str2) {
    string res = mns(str1, multi(str2, div(str1, str2)));
    transform(res);
    remove_lid_zers(res);
    return res;
}


int main() {
    string str1 = "50";
    string str2 = "100";
    string res2 = mod(str1, str2);
    string res1 = div(str1, str2);
    cout << str1 << '\n' << str2 << '\n' << res1 << ' ' << res2 << endl;
}
