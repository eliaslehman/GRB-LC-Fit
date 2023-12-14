import selenium
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
from selenium.common.exceptions import TimeoutException
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from selenium.common.exceptions import NoSuchElementException


from subprocess import CREATE_NO_WINDOW
from selenium.webdriver.chrome.service import Service
from selenium.webdriver import Chrome, ChromeOptions

import sys
import os

from chromedriver_py import binary_path
from selenium.webdriver.chrome.service import Service

params = np.genfromtxt(r"C:\Users\elias\Desktop\BAERI\spectral_index\filtered_data_with_redshift.txt", dtype = str, usecols = (0,11,12))
#print(params)
index=[]
ids = []
#print(params)
#Loop will start here
print(params[0])

for i in range(1,len(params)):

    grbid = params[i][0]
    #if float(params[i][1]) <= 2:
    taerr = np.log(10)*float(params[i][2])*(10**float(params[i][1]))
    tastart = 10**float(params[i][1])-taerr
    tastop = 10**float(params[i][1])+taerr
    print(str(tastart)+' '+str(tastop))
    

#Navigate to webpage

    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=Service(executable_path=binary_path), options=options)

    #driver = webdriver.Chrome()  
    #driver = webdriver.Chrome(executable_path='/User/biagi/Desktop/Support_for_Maria_26August_11October/05December-spectralindex/chromedriver.exe')
    driver.get("http://www.swift.ac.uk/xrt_spectra/")
    field = driver.find_element("name","name")
    field.send_keys(grbid)
    field.send_keys(Keys.RETURN)
    driver.find_element("link text",'Add time-sliced spectrum').click()

#Fill fields

    namefield = driver.find_element("name","name1")
    namefield.send_keys(grbid)
    timefield = driver.find_element("name","time1")
    timefield.send_keys(str(tastart) + ' - ' + str(tastop))
    mode = driver.find_element("name","mode1")
    mode.send_keys('BOTH')
    driver.find_element(By.CLASS_NAME,"button").click()

#Wait for output
#//*[@id="holder"]/p[4]/a
    print("gothere1")
    try:
        #element1 =  WebDriverWait(driver, 14400).until(EC.presence_of_element_located(name = "Time-slicing failed"))
        #driver.quit()
        element = WebDriverWait(driver, 300).until(EC.presence_of_element_located((By.XPATH, '//*[@id="holder"]/h3')))
        if ("Time-slicing  failed" in driver.page_source):
            driver.get("http://www.swift.ac.uk/xrt_spectra/")
            field = driver.find_element("name","name")
            field.send_keys(grbid)
            field.send_keys(Keys.RETURN)
            driver.find_element("link text",'Add time-sliced spectrum').click()
            namefield = driver.find_element("name","name1")
            namefield.send_keys(grbid)
            timefield = driver.find_element("name","time1")
            timefield.send_keys(str(tastart) + ' - ' + str(tastop + 100))
            mode = driver.find_element("name","mode1")
            mode.send_keys('BOTH')
            driver.find_element(By.CLASS_NAME,"button").click()
            print('+100')
            
            element = WebDriverWait(driver, 300).until(EC.presence_of_element_located((By.XPATH, "//*[@id='holder']/h3")))
    
            if ("Time-slicing  failed" in driver.page_source):
                driver.get("http://www.swift.ac.uk/xrt_spectra/")
                field = driver.find_element("name","name")
                field.send_keys(grbid)
                field.send_keys(Keys.RETURN)
                driver.find_element("link text",'Add time-sliced spectrum').click()
                namefield = driver.find_element("name","name1")
                namefield.send_keys(grbid)
                timefield = driver.find_element("name","time1")
                timefield.send_keys(str(tastart) + ' - ' + str(tastop + 1000))
                mode = driver.find_element("name","mode1")
                mode.send_keys('BOTH')
                driver.find_element(By.CLASS_NAME,"button").click()
                print('+1000')
    
                element = WebDriverWait(driver, 300).until(EC.presence_of_element_located((By.XPATH, "//*[@id='holder']/h3")))
    
                if ("Time-slicing  failed" in driver.page_source):
                    driver.get("http://www.swift.ac.uk/xrt_spectra/")
                    field = driver.find_element("name","name")
                    field.send_keys(grbid)
                    field.send_keys(Keys.RETURN)
                    driver.find_element("link text",'Add time-sliced spectrum').click()
                    namefield = driver.find_element("name","name1")
                    namefield.send_keys(grbid)
                    timefield = driver.find_element("name","time1")
                    timefield.send_keys(str(tastart) + ' - ' + str(tastop + 10000))
                    mode = driver.find_element("name","mode1")
                    mode.send_keys('BOTH')
                    driver.find_element(By.CLASS_NAME,"button").click()
                    print('+10000')
    
                    element = WebDriverWait(driver, 300).until(EC.presence_of_element_located((By.XPATH, "//*[@id='holder']/h3")))
    
                    if ("Time-slicing  failed" in driver.page_source):
                        driver.get("http://www.swift.ac.uk/xrt_spectra/")
                        field = driver.find_element("name","name")
                        field.send_keys(grbid)
                        field.send_keys(Keys.RETURN)
                        driver.find_element("link text",'Add time-sliced spectrum').click()
                        namefield = driver.find_element("name","name1")
                        namefield.send_keys(grbid)
                        timefield = driver.find_element("name","time1")
                        timefield.send_keys(str(tastart) + ' - ' + str(tastop + 100000))
                        mode = driver.find_element("name","mode1")
                        mode.send_keys('BOTH')
                        driver.find_element(By.CLASS_NAME,"button").click()
                        print('+100000')
    
                        element = WebDriverWait(driver, 300).until(EC.presence_of_element_located((By.XPATH, "//*[@id='holder']/h3")))
    
                        if ("Time-slicing  failed" in driver.page_source):
                            driver.get("http://www.swift.ac.uk/xrt_spectra/")
                            field = driver.find_element("name","name")
                            field.send_keys(grbid)
                            field.send_keys(Keys.RETURN)
                            driver.find_element("link text",'Add time-sliced spectrum').click()
                            namefield = driver.find_element("name","name1")
                            namefield.send_keys(grbid)
                            timefield = driver.find_element("name","time1")
                            timefield.send_keys(str(tastart) + ' - ' + str(tastop + 1000000))
                            mode = driver.find_element("name","mode1")
                            mode.send_keys('BOTH')
                            driver.find_element(By.CLASS_NAME,"button").click()
                            print('+1000000')
    
                            element = WebDriverWait(driver, 300).until(EC.presence_of_element_located((By.XPATH, "//*[@id='holder']/h3")))
    
        print("gothere3")
        table = driver.find_element(By.XPATH,"//*[@id='holder']/div[3]/table/tbody/tr[4]/td").text
        try:
            table2 =  driver.find_element(By.XPATH,"//*[@id='holder']/div[4]/table/tbody/tr[4]/td").text
            print(grbid)
            print(table)
            print(table2)
            #print(table1)
        
            #Append output to array
            
            index.append(table + " " + table2)
            ids.append(grbid)
        except NoSuchElementException:
            print(grbid)
            print(table)
            #print(table1)
        
            #Append output to array
            
            index.append(table)
            ids.append(grbid)
            driver.quit()
        #table1 = driver.find_element_by_xpath("//*[@id='holder']/div[4]/table/tbody/tr[4]/td").text
        
       
    except TimeoutException as ex:
        print("Exception has been thrown. " + str(ex))
        driver.quit()
    driver.quit()

    #Clean output

no_dec = re.compile(r'[^\d.]+')
print(index)
for i in range(len(index)):
    index[i] = index[i].split()
    length = len(index[i])
    for j in range(length):
        index[i][j] = no_dec.sub('',index[i][j])

#Write to file

indexes = open('indexes_05December2023.txt','w')

for i in range(len(index)):
    indexes.write(str(ids[i])+' ')
    length = len(index[i])
    for j in range(length):
        indexes.write(index[i][j])
        indexes.write(' ')
    indexes.write('\n')

indexes.close()

