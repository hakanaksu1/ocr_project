from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import tempfile
import os
import logging
from google.cloud import vision
from google.cloud.vision_v1 import types
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pdf2image import convert_from_path
import re
import json

app = FastAPI(title="Invoice OCR API")
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def set_google_api_credentials(credential_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

def extract_text_with_bounding_boxes(client, image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The file {image_path} does not exist.")
    
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

   
    if response.error.message:
        raise ValueError(f"Google Vision API Error: {response.error.message}")
    
    return [{'text': text.description, 'bounding_box': text.bounding_poly.vertices} for text in texts[1:]]

def is_nearby(x1, y1, x2, y2, threshold_x, threshold_y):
    return abs(x1 - x2) <= threshold_x and abs(y1 - y2) <= threshold_y

def find_words_in_area(ocr_data, x_start, y_start, region_width, y_bottom_offset, y_top_offset,count=0):
    
    if count==1:
        
        return [
                data['text'] for data in ocr_data
                
                if ((x_start< data['bounding_box'][0].x <= x_start + region_width and
                    y_start  <= data['bounding_box'][2].y <= y_start+7 ))
                    or
                   ((x_start-region_width < data['bounding_box'][0].x <= x_start + region_width and
                    y_start +20   <= data['bounding_box'][2].y <= y_start+y_bottom_offset ))
                    
            ]
    elif count==2:
        return [
                data['text'] for data in ocr_data
                
                if (x_start-2*region_width < data['bounding_box'][0].x <= x_start + region_width and
                    y_start+5  <= data['bounding_box'][2].y <= y_start + y_bottom_offset)
                    
            ]
    else:
        
        return [
                data['text'] for data in ocr_data
                
                if (x_start < data['bounding_box'][0].x <= x_start + region_width and
                    y_start - y_top_offset <= data['bounding_box'][2].y <= y_start + y_bottom_offset)
                    
            ]
    
def find_words_in_line(ocr_data, x_start, y_start, region_width, y_threshold):
    return [
        data['text'] for data in ocr_data
        if (x_start < data['bounding_box'][0].x <= x_start + region_width) and (abs(y_start - data['bounding_box'][2].y) <= y_threshold)
    ][:3]

def find_and_connect_words(ocr_data, keyword_sequences, region_width, region_height_offset, last_difference_value, is_ozet):
    sentences, all_midpoints = [], []

    for keyword_list in keyword_sequences:
        current_sentence, previous_bbox, first_word_coordinates = [], None, None
        for data in ocr_data:
            text = data['text']
            bbox = data['bounding_box']

            if text == keyword_list[0] and not current_sentence:
                current_sentence.append(text)
                previous_bbox = bbox
                first_word_coordinates = (bbox[0].x, bbox[2].y)
                continue

            if current_sentence and len(current_sentence) < len(keyword_list):
                if text == keyword_list[len(current_sentence)]:
                    if is_nearby(previous_bbox[2].x, previous_bbox[2].y, bbox[0].x, bbox[2].y, 20, 5):
                        current_sentence.append(text)
                        previous_bbox = bbox
                        if len(current_sentence) == len(keyword_list):
                            complete_sentence = " ".join(current_sentence)
                            last_word_coordinates = (bbox[2].x, bbox[2].y - 8)
                            sentences.append((complete_sentence, last_word_coordinates))
                            
                            midpoint_y = (first_word_coordinates[1] + last_word_coordinates[1]) / 2
                            all_midpoints.append((last_word_coordinates, midpoint_y))
                            break
                elif text == keyword_list[0]:
                    current_sentence = [text]
                    previous_bbox = bbox
                    first_word_coordinates = (bbox[0].x, bbox[2].y)

    differences = [((mp[0][0], mp[0][1]), all_midpoints[i + 1][1] - mp[1]) for i, mp in enumerate(all_midpoints[:-1])]
    
    if is_ozet and differences:
        differences[0] = (differences[0][0], 50)
    
    if all_midpoints:
        differences.append(((all_midpoints[-1][0][0], all_midpoints[-1][0][1]), last_difference_value))

    return [
        (sentence, " ".join(find_words_in_area(ocr_data, x, y, region_width, diff[1] - region_height_offset, 0)))
        for ((sentence, (x, y)), diff) in zip(sentences, differences)
    ]

def find_tuketimleriniz_and_years(ocr_data):
    sentences = []
    current_year = str(datetime.now().year)
    previous_year = str(datetime.now().year - 1)
    years_to_check = {current_year, previous_year}
    
    tuketimleriniz_bbox = None
    for data in ocr_data:
        if data['text'].upper() == "TÜKETİMLERİNİZ":
            tuketimleriniz_bbox = data['bounding_box']
            break

    if tuketimleriniz_bbox:
        for data in ocr_data:
            text = data['text']
            bbox = data['bounding_box']
            
            if text in years_to_check:
                if is_nearby(tuketimleriniz_bbox[2].x, tuketimleriniz_bbox[2].y, bbox[2].x, bbox[2].y, 300, 300):
                    start_x = bbox[2].x
                    start_y = bbox[2].y
                    
                    toplam_tuketim = find_words_in_area(ocr_data, start_x, start_y, 1000, 10, 0)[:4]
                    
                    gunluk_ortalama_tuketim = find_words_in_area(ocr_data, start_x, start_y, 1000, 10, 0)[4:10]
                    
                    sentences.append((f"{text} toplam tüketim:"," ".join(toplam_tuketim))) 
                    
    return sentences

def find_yekdem_and_check_proximity(ocr_data):
    sentences = []
    for data in ocr_data:
        if data['text'] == "YEKDEM":
            bbox_yekdem = data['bounding_box']
            yekdem_x_right, yekdem_y_bottom = bbox_yekdem[2].x, bbox_yekdem[2].y
            for other_data in ocr_data:
                if other_data['text'] == "DİĞER":
                    if is_nearby(yekdem_x_right, yekdem_y_bottom, other_data['bounding_box'][2].x, other_data['bounding_box'][2].y, 50, 300):
                        words_in_area = find_words_in_area(ocr_data, yekdem_x_right, yekdem_y_bottom, 600, 10, 0)
                        if words_in_area:
                            
                            sentences.append((data['text'], " ".join(words_in_area)))
    
    return sentences

def find_target_words_and_proximity(ocr_data):
    sentences = []
    for data in ocr_data:
        text = data['text'].lower()
        if text in {"endüktif", "kapasitif","gündüz","gece","puant"}:
            bbox_target = data['bounding_box']
            target_x_right, target_y_bottom = bbox_target[2].x, bbox_target[2].y

            if any(
                other_data['text'].lower() == "endeks" and is_nearby(target_x_right, target_y_bottom, other_data['bounding_box'][2].x, other_data['bounding_box'][2].y, 50, 300)
                for other_data in ocr_data
            ):
                words_found = find_words_in_line(ocr_data, target_x_right, target_y_bottom, 600, 10)
                
                if len(words_found) == 3:
                    sentences.append((f"{text.capitalize()} ilk endeks:", words_found[0]))
                    sentences.append((f"{text.capitalize()} son endeks:", words_found[1]))
                    sentences.append((f"{text.capitalize()} fark:", words_found[2]))                                                      
    
    return sentences

def find_energy_information(ocr_data):
    sentences = []
    keywords_list = [["Tek", "Zaman"], ["Elk", ".", "Dağıtım"]]

    for keywords in keywords_list:
        result = []

        for i, data in enumerate(ocr_data):
            if data['text'] == keywords[0]:  
                current_sentence = [data['text']]
                previous_bbox = data['bounding_box']
                keyword_index = 1
                found_sequence = False

                for j in range(i + 1, len(ocr_data)):
                    next_data = ocr_data[j]
                    next_text = next_data['text']
                    next_bbox = next_data['bounding_box']

                    if next_text == keywords[keyword_index] and is_nearby(previous_bbox[2].x, previous_bbox[2].y, next_bbox[0].x, next_bbox[2].y, 50, 300):
                        current_sentence.append(next_text)
                        previous_bbox = next_bbox
                        keyword_index += 1

                        if keyword_index == len(keywords):
                            found_sequence = True
                            rightmost_coordinates = (previous_bbox[2].x, previous_bbox[2].y)
                            result.append((current_sentence, rightmost_coordinates))
                            break

                if found_sequence:
                    break

        for sentence, (x_right, y_bottom) in result:
            words_found = find_words_in_line(ocr_data, x_right, y_bottom, 600, 10)

            if len(words_found) == 3:
                if set(sentence) == {"Tek", "Zaman"}:
                    
                    sentences.append((f"Aktif Toplam: tüketim (kwh)", words_found[0]))
                    sentences.append((f"Aktif Toplam: birim fiyat(tl)", words_found[1]))
                    sentences.append((f"Aktif Toplam: bedel(tl)", words_found[2]))
                elif set(sentence) == {"Elk", ".", "Dağıtım"}:
                    
                    sentences.append((f"Dağıtım bedeli: tüketim (kwh)", words_found[0]))
                    sentences.append((f"Dağıtım bedeli: birim fiyat(tl)", words_found[1]))
                    sentences.append((f"Dağıtım bedeli: bedel(tl)", words_found[2]))
    return sentences


def filter_numbers_and_periods(text):
    
    
    return re.sub(r'[^0-9.]', '', text)

def find_sequence(ocr_data, main_keyword, sub_keywords):
    
    for data in ocr_data:
        if data['text'] == main_keyword:
            current_bbox = data['bounding_box']
            for sub_keyword in sub_keywords:
                next_data = next(
                    (item for item in ocr_data if item['text'] == sub_keyword and 
                     is_nearby(current_bbox[2].x, current_bbox[2].y, item['bounding_box'][0].x, item['bounding_box'][2].y, 50, 15)), 
                    None
                )
                if next_data:
                    current_bbox = next_data['bounding_box']
                else:
                    break
            else:
                return (f"{main_keyword} {' '.join(sub_keywords)}", current_bbox[2].x, current_bbox[2].y)
    return None

def process_keyword_sequences(ocr_data, keyword_sequences):
    
    sentences, all_midpoints = [], []

    for keywords in keyword_sequences:
        index = 0
        while index < len(ocr_data):
            current_sentence, previous_bbox, current_index = [], None, 0
            data = ocr_data[index]
            text, bbox = data['text'], data['bounding_box']
            
            if text == keywords[current_index]:
                current_sentence = [text]
                previous_bbox = bbox
                current_index += 1

                for i in range(index + 1, len(ocr_data)):
                    next_text, next_bbox = ocr_data[i]['text'], ocr_data[i]['bounding_box']
                    
                    if next_text == keywords[current_index] and is_nearby(previous_bbox[2].x, previous_bbox[2].y, next_bbox[0].x, next_bbox[2].y, 50, 15):
                        current_sentence.append(next_text)
                        previous_bbox = next_bbox
                        current_index += 1

                        if current_index == len(keywords):
                            right_mid_x = (next_bbox[1].x + next_bbox[2].x) / 2
                            right_mid_y = (next_bbox[1].y + next_bbox[2].y) / 2
                            sentences.append((f"{' '.join(current_sentence)}", right_mid_x, right_mid_y))
                            all_midpoints.append((right_mid_x, right_mid_y))
                            break
                    else:
                        current_index = 0   
                if current_index != len(keywords):
                    index += 1
                else:
                    break
            else:
                index += 1
    
    return sentences, all_midpoints

def extract_text_in_area(ocr_data, sentences, all_midpoints):
    
    results = []
    for i, (sentence, x_mid, y_mid) in enumerate(sentences):
        y_top_offset = max(y_mid - all_midpoints[i - 1][1] - 25, 0) if i > 0 else 0
        y_bottom_offset = max(all_midpoints[i + 1][1] - y_mid - 13, 0) if i < len(all_midpoints) - 1 else 10

        words_in_area = find_words_in_area(ocr_data, x_mid, y_mid, region_width=400, y_bottom_offset=y_bottom_offset, y_top_offset=y_top_offset)
        if sentence == "FATURA TARİHİ":
            words_in_area = [filter_numbers_and_periods(" ".join(words_in_area))]

        results.append((sentence, " ".join(words_in_area)))

    return results
def find_fatura_number(ocr_data):
    
    for i, data in enumerate(ocr_data):
        text = data['text'].upper()
        if text == "FATURA":
            current_bbox = data['bounding_box']
            
            next_word = ""
            next_bbox = None
            region_width = 300  
            
            if i + 1 < len(ocr_data):
                next_text1 = ocr_data[i + 1]['text'].upper()
                next_bbox1 = ocr_data[i + 1]['bounding_box']
                if next_text1 == "NO" and is_nearby(current_bbox[2].x, current_bbox[2].y, next_bbox1[0].x, next_bbox1[2].y, 50, 15):
                   
                    next_word = next_text1
                    next_bbox = next_bbox1
                    region_width = 300
                elif next_text1 == "SIRA" and i + 2 < len(ocr_data):
                    next_text2 = ocr_data[i + 2]['text'].upper()
                    next_bbox2 = ocr_data[i + 2]['bounding_box']
                    if next_text2 == "NO" and is_nearby(current_bbox[2].x, current_bbox[2].y, next_bbox2[0].x, next_bbox2[2].y, 50, 15):
                        
                        next_word = next_text1 + " " + next_text2
                        next_bbox = next_bbox2
                        region_width = 150
            if next_word and next_bbox:
               
                words_in_area = find_words_in_area(ocr_data, next_bbox[2].x, next_bbox[2].y, region_width=region_width, y_top_offset=15, y_bottom_offset=15)
                return " ".join(words_in_area)
    return ""


def find_endeks_coordinates(ocr_data):
    
    endeks_data = {'endüktif': {'ilk_endeks': [], 'son_endeks': []}, 'kapasitif': {'ilk_endeks': [], 'son_endeks': []},'gündüz': {'ilk_endeks': [], 'son_endeks': []},'gece': {'ilk_endeks': [], 'son_endeks': []},'puant': {'ilk_endeks': [], 'son_endeks': []}}

    for i, data in enumerate(ocr_data):
        text, bbox = data['text'], data['bounding_box']

        if text in ['İlk', 'Son']:
            for next_data in ocr_data[i + 1:]:
                next_text, next_bbox = next_data['text'], next_data['bounding_box']
                
                if next_text == 'Endeks' and is_nearby(bbox[2].x, bbox[2].y, next_bbox[0].x, next_bbox[2].y, 50, 15):
                    x_right, y_bottom = next_bbox[2].x, next_bbox[2].y
                    endeks_value = find_words_in_area(ocr_data, x_right, y_bottom, 300, 8, 5)
                    
                    for other_data in ocr_data:
                        other_text, other_bbox = other_data['text'], other_data['bounding_box']
                        
                        if other_text in ['Endüktif', 'Kapasitif',"Gündüz","Gece","Puant"] and is_nearby(x_right, y_bottom, other_bbox[0].x, other_bbox[2].y, 150, 60):
                            if other_text == 'Endüktif' :
                                key_type = 'endüktif' 
                            elif other_text == 'Kapasitif':
                                key_type = 'kapasitif'
                            elif other_text == 'Gece':
                                key_type = 'gece'
                            elif other_text == 'Puant':
                                
                                key_type = 'puant'
                            else:
                                key_type = 'gündüz'
                                
                            key = 'ilk_endeks' if text == 'İlk' else 'son_endeks'
                            
                            if endeks_value:
                                endeks_data[key_type][key].append((x_right, y_bottom, endeks_value[0]))
                           
                            break
                    break

    return endeks_data

def process_keywords(ocr_data, keywords_dict):
    
    count=0
    results = []
    for keywords, (region_width, diff) in keywords_dict.items():
        main_keyword, *sub_keywords = keywords.split()
        result = find_sequence(ocr_data, main_keyword, sub_keywords)
        if result:
            
            phrase, ref_x, ref_y = result
            y_bottom_offset = diff - 15
            if phrase=="FATURA TUTARI":
                count=1
            elif phrase =="ÖDENECEK TUTAR":
                count=2
            else:
                count=0
            
            words_in_area = find_words_in_area(ocr_data, ref_x, ref_y, region_width=region_width, y_bottom_offset=y_bottom_offset, y_top_offset=5,count=count)
            
            concatenated_words = " ".join(words_in_area)
            
            if phrase == "FATURA TARİHİ" or phrase =="ÖDENECEK TUTAR":
                concatenated_words = re.sub(r'[^0-9.]', '', concatenated_words)

            results.append((phrase, concatenated_words))
    return results

def find_and_combine_keywords(ocr_data, keyword_groups):
    combined_keywords, found_keywords = {}, set()
    for keywords in keyword_groups:
        if tuple(keywords) in found_keywords:
            continue
        current_sequence = []
        for data in ocr_data:
            if data['text'] == keywords[len(current_sequence)]:
                current_sequence.append((data['text'], data['bounding_box']))
                if len(current_sequence) == len(keywords):
                    rightmost_bbox = current_sequence[-1][1]
                    extended_text = find_words_in_area(ocr_data, rightmost_bbox[2].x, rightmost_bbox[2].y, 250, 10, 5)
                    combined_keywords[" ".join([kw[0] for kw in current_sequence])] = " ".join(extended_text)
                    found_keywords.add(tuple(keywords))
                    break
            else:
                current_sequence = []
    return combined_keywords
def process_energy_results(energy_results):
    
    processed_results = []
    try:
        a_str = energy_results['Aktif Tüketim Toplamı ( kWh )']
        b_str = energy_results['Birim Fiyat']
        c_str = energy_results['Dağıtım Bedeli Birim Fiyat']
        d_str = energy_results['Dağıtım Bedeli Tutarı']
    
        
        
        a = round(float(a_str.replace('.', '').replace(',', '.')), 2)
        b = round(float(b_str.replace(',', '.')), 2)
        c = round(float(c_str.replace(',', '.')), 2)
        d = round(float(d_str.replace('.', '').replace(',', '.')), 2)
        
        
        aktif_tuketim_bedeli = round(a * b, 2)
        dagitim_tuketim_toplami = round(d / c, 2)
        
        dagitim_tuketim_toplami_str=str(dagitim_tuketim_toplami)
        dagitim_tuketim_toplami_str=dagitim_tuketim_toplami_str.replace('.', ',')
        
        aktif_tuketim_bedeli_str=str(aktif_tuketim_bedeli)
        aktif_tuketim_bedeli_str=aktif_tuketim_bedeli_str.replace('.', ',')
        
        
        processed_results = [
            ('Aktif Tüketim Toplamı ( kWh )', a_str),
            ('Aktif Tüketim Birim Fiyat', b_str),
            ('Aktif Tüketim Bedeli (TL)', aktif_tuketim_bedeli_str),
            ('Dağıtım Tüketim Toplamı ( kWh )', f"{dagitim_tuketim_toplami:.2f}"),
            ('Dağıtım Bedeli Birim Fiyat', c_str),
            ('Dağıtım Bedeli Tutarı (TL)', d_str)
        ]
      
    except KeyError as ke:
       pass
    except ValueError as ve:
       pass
    
    return processed_results
def find_carpan_with_related_words(ocr_data, y_offset=5, x_offset=200, search_width=200, search_height=50):
    
    result = []


    for word in ocr_data:
        if word['text'].strip().lower() == "çarpan":
            carpan = word
            break
    else:
        
        return result
    
   
    sag_alt = carpan['bounding_box'][2]
    sag_alt_x = sag_alt.x
    sag_alt_y = sag_alt.y
    
    
    target_x = sag_alt_x + x_offset
    target_y = sag_alt_y - y_offset
    
    
    x_min, x_max = target_x, target_x + search_width
    y_min, y_max = target_y - search_height, target_y + search_height
    
    
    related_word = None
    for word in ocr_data:
        word_center_x = (word['bounding_box'][0].x + word['bounding_box'][2].x) / 2
        word_center_y = (word['bounding_box'][0].y + word['bounding_box'][2].y) / 2
    
        if x_min <= word_center_x <= x_max and y_min <= word_center_y <= y_max:
            
            related_word = word['text'].strip().replace(".", "")
            related_word = related_word.strip().replace(",", ".")
            break  
    if related_word:
        result.append(("Çarpan", related_word))
    
    return result
def convert_pdf_to_images(pdf_path):
    
    images = convert_from_path(pdf_path)
    image_paths = [f"temp_page_{i + 1}.jpg" for i, image in enumerate(images)]
    for image_path, image in zip(image_paths, images):
        image.save(image_path, 'JPEG')
    return image_paths
def parse_endeks_strings(endeks_data, key_value_pairs=None):
    
    if key_value_pairs is None:
        key_value_pairs = []
    
    key_types = ['endüktif', 'kapasitif', 'gündüz', 'gece', 'puant']

    for key_type in key_types:
        

        ilk_list = endeks_data[key_type].get('ilk_endeks', [])
        son_list = endeks_data[key_type].get('son_endeks', [])

      

        for ilk, son in zip(ilk_list, son_list):
            
            try:
                ilk_val = float(ilk[2].replace(',', '.'))
                son_val = float(son[2].replace(',', '.'))
                fark = round(abs(ilk_val - son_val), 2)

                key_value_pairs.append((f"{key_type.capitalize()} İlk Endeks", ilk[2]))
                key_value_pairs.append((f"{key_type.capitalize()} Son Endeks", son[2]))
                key_value_pairs.append((f"{key_type.capitalize()} Fark", f"{fark:.2f}".replace('.', ',')))
            except ValueError:
                continue

    return key_value_pairs

def extract_invoice_details(data):
    result = {}
    
    
    for key, value in data:
        if "Fatura Sıra No" in key or "FATURA NO" in key:
            result['invoiceNumber'] =value  
        elif 'SÖZLEŞME HESAP NO' in key.upper() or 'SÖZLEŞME NO' in key.upper():
            result['contractNumber'] = value
        
        
        elif 'TESİSAT NO / TEKİL KOD' in key.upper():
            result['installationNumber'] = value
        
        
        elif 'MÜŞTERİ ADI SOYADI' in key.upper():
            result['customerName'] = value
        
       
        elif 'VERGİ DAİRESİ - VKN' in key.upper():
         
            if '/' in value:
                parts = value.split('/', 1)
            elif '-' in value:
                parts = value.split('-', 1)
            else:
                parts = [value]
            
            
            if len(parts) >= 2:
                result['taxOffice'] = parts[0].strip()
                result['vkn'] = parts[1].strip()
            else:
                result['taxOffice'] = value.strip()
                result['vkn'] = ""  
        
        
        elif 'TÜKETİCİ GRUBU' in key.upper() :
            result['companySubscription'] = value 
        elif 'ABONE GRUBU' in key.upper():
            result['abone'] = value  
        elif 'TÜKETİCİ SINIFI' in key.upper():
            result['sinif'] = value 
        elif "TESİSAT ADRESİ" in key.upper() :
            result['address'] = value
        elif "FATURA DÖNEMİ / TARİHİ" in key.upper() or "FATURA TARİHİ" in key.upper():
            result["invoiceDate"] = value
        elif "EIC KODU" in key.upper() or "ETSO / EIC" in key.upper():
            result["etso"] = value
        elif 'İlk / Son Okuma Tarihi' in key:
            
            dates = value.split(" / ")
            if len(dates) == 2:
                result['firstReadingDate'] = dates[0].strip()  
                result['lastReadingDate'] = dates[1].strip()   
        elif "İlk Okuma Tarihi" in key:
            result['firstReadingDate'] =value
        elif "Son Okuma Tarihi" in key:
            result['lastReadingDate'] =value
        elif "2023 toplam tüketim" in key:
            consumption = value.split(" ( ")
            
            
            result['lastYearConsumption'] =float(consumption[0] .replace('.', '').replace(',', '.'))
        elif "Çarpan" in key:
            result["multiplier"] = float(value)
        elif "2024 toplam tüketim" in key:
            consumption = value.split(" ( ")
            
            
            result['thisYearConsumption'] =float(consumption[0] .replace('.', '').replace(',', '.'))
        elif "Önceki Yıl Tüketim ( kWh )" in key:
             result['lastYearConsumption'] =float(value.replace('.', '').replace(',', '.'))
        elif "Cari Yıl Tüketim ( kWh )" in key:
             result['thisYearConsumption'] =round(float(value.replace('.', '').replace(',', '.')),2)
             
        elif "Aktif Toplam: tüketim (kwh)" in key or "Aktif Tüketim Toplamı ( kWh )" in key:
             result['totalActiveConsumption'] =round(float(value.replace('.', '').replace(',', '.')), 2)     
        elif "Aktif Tüketim Bedeli (TL)" in key or "Aktif Toplam: bedel(tl)" in key:
             result['totalActiveEnergyCost'] =round(float(value.replace('.', '').replace(',', '.')), 2)
             
        elif "Dağıtım Bedeli Tutarı (TL)" in key or "Dağıtım bedeli: bedel(tl)" in key:
              result['distributionCost'] =round(float(value.replace('.', '').replace(',', '.')), 2) 
              
        elif "Endüktif ilk endeks" in key or "Endüktif İlk Endeks" in key:
              result['inductiveInitialindex'] =round(float(value.replace('.', '').replace(',', '.')), 2)     
        elif "Endüktif son endeks" in key or "Endüktif Son Endeks" in key:
               result['inductiveLastindex'] =round(float(value.replace('.', '').replace(',', '.')), 2)     
        elif "Endüktif fark" in key or "Endüktif Fark" in key:
           
            
            result['inductiveDifference'] =round(float(value.replace('.', '').replace(',', '.')),2)
               
        elif "Kapasitif ilk endeks" in key or "Kapasitif İlk Endeks" in key:
              result['capacitiveInitialindex'] =round(float(value.replace('.', '').replace(',', '.')), 2)      
        elif "Kapasitif son endeks" in key or "Kapasitif Son Endeks" in key:
               result['capacitiveLastindex'] =round(float(value.replace('.', '').replace(',', '.')), 2)     
        elif "Kapasitif fark" in key or "Kapasitif Fark" in key:
               result['capacitiveDifference'] =round(float(value.replace('.', '').replace(',', '.')),2)
       
        elif "Gündüz ilk endeks" in key or "Gündüz İlk Endeks" in key:
              result['daytimeInitialindex'] =round(float(value.replace('.', '').replace(',', '.')), 2)      
        elif "Gündüz son endeks" in key or "Gündüz Son Endeks" in key:
               result['daytimeLastindex'] =round(float(value.replace('.', '').replace(',', '.')), 2)     
        elif "Gündüz fark" in key or "Gündüz Fark" in key:
               result['daytimeDifference'] =round(float(value.replace('.', '').replace(',', '.')),2) 
        elif "Gece ilk endeks" in key or "Gece İlk Endeks" in key:
              result['nightInitialindex'] =round(float(value.replace('.', '').replace(',', '.')), 2)       
        elif "Gece son endeks" in key or "Gece Son Endeks" in key:
               result['nightLastindex'] =round(float(value.replace('.', '').replace(',', '.')), 2)      
        elif "Gece fark" in key or "Gece Fark" in key:
               result['nightDifference'] =round(float(value.replace('.', '').replace(',', '.')),2)
        elif "Puant ilk endeks" in key or "Puant İlk Endeks" in key:
              result['puantInitialindex'] =round(float(value.replace('.', '').replace(',', '.')), 2)     
        elif "Puant son endeks" in key or "Puant Son Endeks" in key:
               result['puantLastindex'] =round(float(value.replace('.', '').replace(',', '.')), 2)     
        elif "Puant fark" in key or "Puant Fark" in key:
               result['puantDifference'] =round(float(value.replace('.', '').replace(',', '.')),2)   
        elif "YEKDEM" in key or "Yekdem Fark Bedeli" in key:
               result['yekdemCost'] =round(float(value.replace('.', '').replace(',', '.')), 2)
        elif   "KDV % 20" in key:
            btvcost = value.split(" ) ")
            
            result['btvCost'] =round(float(btvcost[1].replace('.', '').replace(',', '.')), 2) 
        elif "Katma Değer Vergisi ( % 18 , % 20 )" in key: 
           
            
            result['btvCost'] =round(float(value.replace('.', '').replace(',', '.')), 2) 
        elif "FATURA TUTARI" in key :
            value=re.sub(r'[^0-9.,]', '', value)     
            result['totalInvoiceAmount'] =round(float(value.replace('.', '').replace(',', '.')), 2) 
        elif "ÖDENECEK TUTAR" in key :
                  result['totalAmountDue'] =round(float(value.replace('.', '').replace(',', '.')), 2) 
        elif "SON ÖDEME TARİHİ" in key :
                  result['dueDate'] =value   
         
        
   
   
    if 'abone' in result and 'sinif' in result:
        result['companySubscription'] = result['abone'] + " " + result['sinif']
        del result['abone']
        del result['sinif']
        
    result['inductiveConsumption']= result["multiplier"]*result['inductiveDifference']      
    result['capacitiveConsumption']= result["multiplier"]*result['capacitiveDifference']       
    return result


def process_file(file_path, credential_path):
   
    set_google_api_credentials(credential_path)
    
   
    supported_extensions = ['.pdf', '.jpeg', '.jpg', '.png']
    
   
    _, file_extension = os.path.splitext(file_path.lower())
    
   
    if file_extension not in supported_extensions:
        raise ValueError(f"Unsupported file extension: {file_extension}. Supported formats are: {', '.join(supported_extensions)}.")
    
    
    client = vision.ImageAnnotatorClient()
    if file_extension == '.pdf':
        image_paths = convert_pdf_to_images(file_path)
    else:
        image_paths = [file_path]
    
    
    result = process_image_file(client, image_paths)
    
    return result





def process_image_file(client, image_paths):
    try:
        
        first_page_ocr_data = extract_text_with_bounding_boxes(client, image_paths[0])
        all_words = {data['text'].strip().lower() for data in first_page_ocr_data}
        
       
        if "enerjisa" in all_words:
            result = main_process_enerjisa(client, image_paths)  
        elif "gediz" in all_words:
            result = main_process_gediz(client, image_paths)
        else:
            raise ValueError("Fatura türü tanınamadı.")
        
        return result
    except Exception as e:
        logger.error(f"Error in process_image_file: {e}")
        raise e

def main_process_enerjisa(client, image_paths):
    
    
    keyword_data = [
        ([
            ["SÖZLEŞME", "HESAP", "NO"],
            ["TESİSAT", "NO", "/", "TEKİL", "KOD"],
            ["MÜŞTERİ", "NO"],
            ["MÜŞTERİ", "ADI", "SOYADI"],
            ["VERGİ", "DAİRESİ", "-", "VKN"],
            ["TÜKETİCİ", "GRUBU"],
            ["TESİSAT", "ADRESİ"],
            ["OTOMATİK", "ÖDEME", "TALİMATI", ":"],
            ["FATURA", "DÖNEMİ", "/", "TARİHİ"]
        ], 450, 6, 17, False),
        ([
            ["İlk", "/", "Son", "Okuma", "Tarihi"],
            ["Gerilim", "Trafo", "Oranı"],
            ["ETSO", "/", "EIC"]
        ], 420, 83, 100, False),
        ([
            ["FATURA", "TUTARI"],
            ["ÖDENECEK", "TUTAR"],
            ["SON", "ÖDEME", "TARİHİ"]
        ], 450, 6, 17, True),
        ([
            ["Elekt",".", "ve", "Hvg", ".","Tük" ,".", "Ver", "."],
            ["KDV", "%", "20"]
        ], 600, 6, 26, False)
    ]
    
   
    all_sentences = []
   
    
   
    for image_path in image_paths:
        ocr_data = extract_text_with_bounding_boxes(client, image_path)
        fatura_sira_no = find_fatura_number(ocr_data)
        if fatura_sira_no:
            all_sentences.append(("Fatura Sıra No:", fatura_sira_no))  
   
        for keywords, width, height_offset, last_diff, is_ozet in keyword_data:
            sentences = find_and_connect_words(ocr_data, keywords, width, height_offset, last_diff, is_ozet)
            all_sentences.extend(sentences)   
   
        all_sentences.extend(find_yekdem_and_check_proximity(ocr_data)) 
        all_sentences.extend(find_tuketimleriniz_and_years(ocr_data))
        all_sentences.extend(find_target_words_and_proximity(ocr_data))
        all_sentences.extend(find_energy_information(ocr_data))
        all_sentences.extend(find_carpan_with_related_words(ocr_data, y_offset=5, x_offset=200, search_width=400, search_height=5))
    
    
    all_sentences.extend(sentences)
    
    all_sentences = [sentence for sentence in all_sentences if sentence]
    
    combined_results = cleaned_list = [(key.rstrip(":").strip(), value.lstrip(":").strip()) for key, value in all_sentences]
     
    
    result=extract_invoice_details(combined_results)
    
    return result

def main_process_gediz(client, image_paths):
    
    
    
    customer_info_keywords = [
        ["SÖZLEŞME", "NO"], ["TESİSAT", "NO", "/", "TEKİL", "KOD"], ["İŞLETME", "ADI"],
        ["MÜŞTERİ", "ADI", "SOYADI"], ["VERGİ", "DAİRESİ", "-", "VKN", "/", "TCKN"],
        ["ABONE", "GRUBU"], ["TÜKETİCİ", "SINIFI"], ["TESİSAT", "ADRESİ"], 
        ["FATURA", "TARİHİ"], ["FATURA", "KODU"]
    ]
    
    genel_bilgiler_keywords = {
        "EIC Kodu": (300, 20),
        "İlk Okuma Tarihi": (300, 20),
        "Son Okuma Tarihi": (300, 20)
    }
    
    fatura_bilgileri_keywords = {
        "FATURA TUTARI": (600, 100),
        "SON ÖDEME TARİHİ": (400, 20),
        "ÖDENECEK TUTAR":(220,100)
    }
    
    vergiler_keywords = {
        "Elektrik Tüketim Vergisi": (300, 20),
        "Katma Değer Vergisi ( % 18 , % 20 )": (300, 20),
        "Yekdem Fark Bedeli": (300, 20)
    }

    tuketim_bilgileri_keywords = {
        "Önceki Yıl Tüketim ( kWh )": (200, 20),
        "Cari Yıl Tüketim ( kWh ) ": (200, 20)
    }

    
    energy_keywords = [
        ['Aktif', 'Tüketim', 'Toplamı', '(', 'kWh', ')'],
        ['Birim', 'Fiyat'],
        ['Dağıtım', 'Bedeli', 'Birim', 'Fiyat'],
        ['Dağıtım', 'Bedeli', 'Tutarı']
    ]
    
    for image_path in image_paths:
        ocr_data = extract_text_with_bounding_boxes(client, image_path)
    all_customer_info_results = []
    all_genel_bilgiler_results = []
    all_fatura_bilgileri_results = []
    all_vergiler_results = []
    all_tuketim_bilgileri_results = []
    all_aktif_tuketim_results = []
    all_carpan_result=[]

    results = {}  

    for image_path in image_paths:
        ocr_data = extract_text_with_bounding_boxes(client, image_path)
        carpan_result=find_carpan_with_related_words(ocr_data, y_offset=5, x_offset=200, search_width=200, search_height=5)
        all_carpan_result.extend(carpan_result)
        
        customer_sentences, all_midpoints = process_keyword_sequences(ocr_data, customer_info_keywords)
        customer_info_results = extract_text_in_area(ocr_data, customer_sentences, all_midpoints)
        all_customer_info_results.extend(customer_info_results)

        
        fatura_no = find_fatura_number(ocr_data)
        if fatura_no:
            all_customer_info_results.append(("FATURA NO:", fatura_no))

       
        genel_bilgiler_results = process_keywords(ocr_data, genel_bilgiler_keywords)
        all_genel_bilgiler_results.extend(genel_bilgiler_results)
        fatura_bilgileri_results = process_keywords(ocr_data, fatura_bilgileri_keywords)
        all_fatura_bilgileri_results.extend(fatura_bilgileri_results)
        vergiler_results = process_keywords(ocr_data, vergiler_keywords)
        all_vergiler_results.extend(vergiler_results)
        tuketim_bilgileri_results = process_keywords(ocr_data, tuketim_bilgileri_keywords)
        all_tuketim_bilgileri_results.extend(tuketim_bilgileri_results)
        energy_results = find_and_combine_keywords(ocr_data, energy_keywords)
        processed_energy_results = process_energy_results(energy_results)
        results.update(processed_energy_results)
        results_list_energy = [(key, value) for key, value in results.items()]
        endeks_data = find_endeks_coordinates(ocr_data)
        parsed_aktiftuketim_results = parse_endeks_strings(endeks_data,all_aktif_tuketim_results)
        
        

        
    combined_results = (
    all_customer_info_results + 
    results_list_energy + 
    all_genel_bilgiler_results + 
    all_fatura_bilgileri_results + 
    all_vergiler_results + 
    all_tuketim_bilgileri_results + 
    parsed_aktiftuketim_results+
    all_carpan_result
    )
    
    result=extract_invoice_details(combined_results)
    
    return result



@app.post("/process_invoice/", response_class=HTMLResponse)
async def process_invoice(request: Request, file: UploadFile = File(...)):
    
    supported_extensions = ['.pdf', '.jpeg', '.jpg', '.png']
    
    filename = file.filename
    _, file_extension = os.path.splitext(filename.lower())
    
    if file_extension not in supported_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file extension: {file_extension}. Supported formats are: {', '.join(supported_extensions)}."
        )
    
  
    credential_path = 'google_api_path.json'  
    
    try:
       
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            tmp.write(await file.read())
            temp_file_path = tmp.name
        logger.info(f"File saved to temporary path: {temp_file_path}")

       
        result = process_file(temp_file_path, credential_path)

        
        os.remove(temp_file_path)
        logger.info(f"Temporary file removed: {temp_file_path}")

       
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": result,      
            "filename": filename,  
            "error": None
        })

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Temporary file removed due to ValueError: {temp_file_path}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": None,
            "filename": filename,
            "error": str(ve)
        })
    except Exception as e:
        logger.error(f"Exception: {e}")
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Temporary file removed due to Exception: {temp_file_path}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": None,
            "filename": filename,
            "error": str(e)
        })

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": None,
        "filename": None,
        "error": None
    })