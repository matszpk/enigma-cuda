/* This Source Code Form is subject to the terms of the Mozilla Public */
/* License, v. 2.0. If a copy of the MPL was not distributed with this */
/* file, You can obtain one at http ://mozilla.org/MPL/2.0/.           */
/* Copyright (c) 2017 Alex Shovkoplyas VE3NEA                          */

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#ifndef _WIN32
#include <dirent.h>
#include <unistd.h>
#endif
#ifdef _WIN32
#include <Windows.h>
#endif
#include <ctime>
#include "util.h"


std::vector<int8_t> TextToNumbers(const string & text)
{
	std::vector<int8_t> numbers;
	numbers.resize(text.length());
	for (int i = 0; i < numbers.size(); ++i) numbers[i] = ToNum(text[i]);
	return numbers;
}

string NumbersToText(const std::vector<int8_t> numbers)
{
	string text;
	text.resize(numbers.size());
	for (int i = 0; i < text.size(); ++i)	text[i] = ToChar(numbers[i]);
	return text;
}

string LoadTextFromConsole()
{
	std::stringstream ss;
	char c;
	while (std::cin >> c) ss << c;
	return ss.str();
}

string LoadTextFromFile(const string & file_name)
{
	std::ifstream fs(file_name);
	if (!fs) throw std::runtime_error(string("File read failed: ") +
        GetAbsolutePath(file_name));
	std::stringstream ss;
	ss << fs.rdbuf();
	return ss.str();
}

void SaveTextToFile(const string & text, const string & file_name)
{
  std::ofstream fs(file_name);
  if (!fs) throw std::runtime_error(string("File write failed: ") +
    GetAbsolutePath(file_name));
  fs << text;
}

string LettersFromText(const string & text)
{
  string result;
  result = text;
  int dst = 0;
  for (int src = 0; src < result.length(); ++src)
  {
    if (result[src] >= 'A' && result[src] <= 'Z')
      result[dst++] = result[src];
    else if (result[src] >= 'a' && result[src] <= 'z')
      result[dst++] = toupper(result[src]);
  }

  result.resize(dst);
  return result;
}

string LettersAndSpacesFromText(const string & text)
{
    string s = LowerCase(s);
    for (int i = 0; i < s.length(); ++i)
      if (s[i] < 'A' || s[i] > 'Z') s[i] = ' ';

    int dst = 0;
    for (int src = 1; src < s.length(); ++src)
      if (s[src - 1] != ' ' || s[src] != ' ') s[dst++] = s[src];
    return s.substr(0, dst);
}

string GetAbsolutePath(const string & file_name)
{
#ifndef _WIN32
        char* out = realpath(file_name.c_str(), NULL);
        string outstr;
        try {
        outstr = out;
        } catch(...)
        { free(out); throw; }
        free(out);
        return outstr;
#else
	char buf[256];
	char* lppPart;
	GetFullPathNameA((char*)file_name.c_str(),
		256, &buf[0], &lppPart); (file_name);
	return buf;
#endif
}

extern const char *__progname;
string GetExeDir()
{
#ifndef _WIN32
    string result = __progname;
    result = result.substr(0, result.find_last_of("/") + 1);
    return result;
#else
  char buf[1024] = { 0 };
  DWORD ret = GetModuleFileNameA(NULL, buf, sizeof(buf));
  string result = buf;
  result = result.substr(0, result.find_last_of("\\/") + 1);
  return result;
#endif
}

string TimeString()
{
  time_t t = std::time(nullptr);
  tm tt;
#ifndef _WIN32
  localtime_r(&t, &tt);
#else
  localtime_s(&tt, &t);
#endif

  std::ostringstream os;
  os << std::put_time(&tt, "%Y-%m-%d %H:%M:%S");
  return os.str();  
}

#ifndef __linux
string TimeDiffString(myclk_t clock)
{
  int seconds = clock / 1000000U;
#else
string TimeDiffString(clock_t clock)
{
  int seconds = clock / CLOCKS_PER_SEC;
#endif
  int s = seconds % 60;
  int m = (seconds / 60) % 60;
  int h = (seconds / 3600) % 24;
  int d = seconds / 86400;

  std::stringstream ss;
  if (d > 0) ss << d << "d ";
  ss << std::setfill('0') << std::setw(2) << h << ':';
  ss << std::setfill('0') << std::setw(2) << m << ':';
  ss << std::setfill('0') << std::setw(2) << s;
  return ss.str();
}

string LowerCase(const string & text)
{
  string result = text;
  for (int i = 0; i < result.length(); ++i) result[i] = tolower(result[i]);
  return result;
}

string UpperCase(const string & text)
{
  string result = text;
  for (int i = 0; i < result.length(); ++i) result[i] = toupper(result[i]);
  return result;
}

std::vector<string> ListFilesInDirectory(const string & directory)
{
#ifndef _WIN32
    DIR* dir = opendir(directory.c_str());
    std::vector<string> result;
    struct dirent* dent;
    while ((dent = readdir(dir))!=NULL)
        result.push_back(dent->d_name);
    closedir(dir);
    return result;
#else
  WIN32_FIND_DATAA FindData;
  std::vector<string> result;

  HANDLE hFind = FindFirstFileA((directory + "*.*").c_str(), &FindData);
  do 
  { 
    if (string(FindData.cFileName) != "." && string(FindData.cFileName) != "..")
      result.push_back(FindData.cFileName); 
  } 
  while (FindNextFileA(hFind, &FindData));
  FindClose(hFind);

  return result;
#endif
}

void TextToClipboard(const std::string & text)
{
#ifdef _WIN32
  OpenClipboard(GetDesktopWindow());
  EmptyClipboard();
  HGLOBAL h = GlobalAlloc(GMEM_MOVEABLE, text.size() + 1);
  if (!h) { CloseClipboard(); return; }
  memcpy(GlobalLock(h), text.c_str(), text.size() + 1);
  GlobalUnlock(h);
  SetClipboardData(CF_TEXT, h);
  CloseClipboard();
  GlobalFree(h);
#endif
}

bool FileExists(const string & file_name) 
{
  std::ifstream f(file_name.c_str());
  return f.good();
}

string Trim(const string & str)
{
  const string whitespace = " \t\r\n";
  int b, e;
  for (b = 0; b < str.length(); ++b) if (whitespace.find(str[b]) == string::npos)  break;
  for (e = str.length(); e > 0; --e) if (whitespace.find(str[e-1]) == string::npos)  break;
  return e > b ? str.substr(b, e-b) : "";
}
