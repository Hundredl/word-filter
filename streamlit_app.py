import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode
from io import StringIO
from utils import generate_frequency_table

# Set page layout to wide
st.set_page_config(layout="wide")

# Streamlit interface
st.title("Word Filter Assistant")
st.write("This app can help you process your text and highlight words you might not know, along with example sentences.")


passage = '''
Evolution of the Flowering Plants
Many aspects of the history of flowering plants (angiosperms) remain mysterious. Evidence of the earliest angiosperms comes from fossilized leaves, stems, fruits, ^ pollen, and, very rarely, flowers. In addition, there has been much study of modern plant morphology (structure) and genetics in order to determine which living species might be most closely related to the ancient ancestors of angiosperms. Despite intensive efforts for over 200 years, scientists have still not reached consensus on which type of plant was the ancestor to the angiosperms, and when and where the angiosperms first evolved. Indeed, Charles Darwin himself called the origin of the flowering plants an "abominable mystery."



What type of plant was the ancestor to the angiosperms? Most botanists now agree that the flowering plants are monophyletic in origin, meaning that they evolved from a common ancestor. Some paleontologists have suggested that the common ancestor may have been a type of cycad (palmlike tropical plants). Other paleontologists maintain that the angiosperms may have evolved from seed-bearing ferns. Finally, analysis of the morphological traits of some primitive living plants suggests that the ancestor may have been related to the modern pines. The question of angiosperm ancestry remains unresolved.



The time and place of the first appearance of flowering plants have long been a topic of great interest. There is good fossil evidence that early angiosperms, including a number resembling modern magnolias, were present in the Early Cretaceous geologic period (more than 100 million years ago). Angiosperms became increasingly abundant during this period. Between 100 million and 65 million years ago, a period known as the Late Cretaceous, angiosperms increased from less than 1 percent of flora (plant life) to well over 50 percent. Many of the modern plant families appeared during this time period. In the Early Tertiary period which followed: angiosperms increased to comprise 90 percent or more of Earth's total flora. Where did these successful plants first originate and spread from?



Analysis of the fossil leaf structure and geographic distribution of the earliest Cretaceous angiosperms has led many biogeographers to conclude that they evolved in the tropics and then migrated poleward. It is known that angiosperms did not become dominant in the high latitudes until the Late Cretaceous. Paleontologists have recovered fossil angiosperm leaves, stems, and pollen from Early Cretaceous deposits in eastern South America and western Africa. These two continents were joined together as part of Gondwanaland, one of two supercontinents that existed at that time. The locations of these early angiosperm finds would have been close to the equator during the Early Cretaceous and are conformable with a model by which angiosperms spread from the tropics poleward.



Not all botanists agree with an African-South American center for the evolution and dispersal of the angiosperms, pointing out that many of the most primitive forms of flowering plants are found in the South Pacific, including portions of Fiji, New Caledonia, New Guinea, eastern Australia, and the Malay Archipelago. Recent genetic research has identified the rare tropical shrub Amborella as being the living plant most closely related to the ancient ancestor of all the angiosperms. This small shrub, which has tiny yellow-white flowers and red fruit; is found only on New Caledonia, a group of islands in the South Pacific. Many botanists conclude that the best explanation for the large numbers of primitive living angiosperms in the South Pacific region is that this is where the flowering plants first evolved and these modern species are relics of this early evolution. Comparisons of the DNA of Amborella and many hundreds of species of flowering plants suggest that the first angiosperm arose and the development of separate species occurred about 135 million years ago.



Recently discovered fossils complicate our understanding of the origin of the angiosperms even further. Paleontologists from China have found beautifully preserved fossils of an angiosperm plant, including flowers and seeds, in Jurassic period deposits from China. The site, which is about 130 million years old, is near modern Beijing. The new fossil plant found at the site is now the oldest known angiosperm. The age of the fossils and the very primitive features of the flowers have led the discoverers to suggest that the earliest flowering plants may have evolved in northern Asia.
'''

# Text input area
search_chinese_meaning = st.checkbox("Search Chinese Meaning", value=False)
input_text = st.text_area("Your passage", passage)
analysis = st.button("Analysis", use_container_width=True)
if analysis:
    with st.spinner('Processing...'):
        df = generate_frequency_table(input_text, search_chinese_meaning)
    st.dataframe(df, use_container_width=True,)