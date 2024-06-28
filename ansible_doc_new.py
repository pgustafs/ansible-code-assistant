from pathlib import Path
from bs4 import BeautifulSoup
from io import StringIO
import requests
import pandas as pd
from langchain_community.document_loaders import TextLoader

class AnsibleDocLoader:
    def __init__(self, raw_txt_docs_folder):
        self.raw_txt_docs_folder = raw_txt_docs_folder

    def save_file(self, text, name):
        """Save text content to a file in the specified path."""
        folder = Path(self.raw_txt_docs_folder)
        folder.mkdir(parents=True, exist_ok=True)
        file_path = folder / f"{name}.txt"
        file_path.write_text(text)

    @staticmethod
    def flatten_table(table):
        """Flatten a pandas DataFrame to a string."""
        flattened_text = ""
        for _, row in table.iterrows():
            row_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
            flattened_text += row_text + ". "
        return flattened_text.strip()

    @staticmethod
    def get_module_name(url):
        """Extract the module name from a URL."""
        module_name = url.rsplit('/', 1)[-1]
        if module_name.endswith('.html'):
            module_name = module_name[:-5]
        return module_name

    @staticmethod
    def fetch_and_parse_url(url):
        """Fetch and parse the HTML content of a URL."""
        response = requests.get(url)
        html_content = response.content
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup

    def process_tables(self, soup):
        """Replace HTML tables with flattened text."""
        for table in soup.find_all('table'):
            df = pd.read_html(StringIO(str(table)))[0]

            # Extract data from specific classes
            combined_data = []
            choices_data = []
            default_data = []

            for row in table.find_all('tr')[1:]:  # Skip the header row
                option_title = row.find(class_='ansible-option-title').get_text(strip=True) if row.find(class_='ansible-option-title') else ""
                option_type = row.find(class_='ansible-option-type').get_text(strip=True) if row.find(class_='ansible-option-type') else ""
                
                # Extract choices entries
                choices_elements = row.select('li > p > code.ansible-option-choices-entry')
                choices = [entry.get_text(strip=True) for entry in choices_elements]

                # Extract default values
                default_elements = row.find_all(class_='ansible-option-default docutils literal notranslate')
                default_values = [entry.get_text(strip=True) for entry in default_elements]

                combined_data.append([option_title, option_type])
                choices_data.append(', '.join(choices))  # Join multiple choices with a comma
                default_data.append(', '.join(default_values))  # Join multiple default values with a comma if necessary

            # Convert extracted data to DataFrames
            combined_df = pd.DataFrame(combined_data, columns=['Option Title', 'Option Type'])
            choices_df = pd.DataFrame(choices_data, columns=['Option Choices'])
            default_df = pd.DataFrame(default_data, columns=['Default Values'])
            
             # Drop original columns if they exist
            for column in ['Parameter', 'Comments']:
                if column in df.columns:
                    df.drop(column, axis=1, inplace=True)

            # Add the new columns to the DataFrame
            df['module parameter'] = combined_df['Option Title']
            df['attribute value'] = choices_df['Option Choices']
            df['parameter value type'] = combined_df['Option Type']
            df['parameter default value'] = default_df['Default Values']

            table_str = self.flatten_table(df)
            table.replace_with(table_str)
        return soup

    @staticmethod
    def extract_text_content(soup):
        """Extract text content from the parsed HTML soup."""
        content = soup.find('div', {'class': 'document'})
        return content.get_text()

    def load(self, urls):
        """Scrape Ansible module documentation from a list of URLs."""
        all_documents = []
        for url in urls:
            module_name = self.get_module_name(url)
            soup = self.fetch_and_parse_url(url)
            soup = self.process_tables(soup)
            text_content = self.extract_text_content(soup)
            self.save_file(text_content, module_name)
        
        directory = Path(self.raw_txt_docs_folder)
        for file_path in directory.glob('*.txt'):
            loader = TextLoader(str(file_path))
            documents = loader.load()
            all_documents.extend(documents)
        return all_documents

class LinkExtractor:
    def __init__(self, url, base_url):
        """
        Initialize the LinkExtractor with a URL and a base URL.

        Parameters:
        url (str): The URL of the webpage to fetch.
        base_url (str): The base URL to prepend to relative links.
        """
        self.url = url
        self.base_url = base_url
        self.soup = None

    def fetch_content(self):
        """
        Fetch and parse the content of the webpage.
        """
        response = requests.get(self.url)
        web_content = response.content
        self.soup = BeautifulSoup(web_content, 'html.parser')

    def extract_links_from_section(self, section_id):
        """
        Extract links from a specific section.

        Parameters:
        section_id (str): The ID of the section to extract links from.

        Returns:
        list: A list of full URLs without fragments.
        """
        if self.soup is None:
            raise ValueError("Content not fetched. Call fetch_content() first.")

        links = []
        section = self.soup.find(id=section_id)
        if section:
            for link in section.find_all('a', href=True):
                full_url = self.base_url + link['href']
                full_url = full_url.split('#')[0]
                links.append(full_url)
        return links

    def extract_links_from_sections(self, section_ids):
        """
        Extract links from multiple sections.

        Parameters:
        section_ids (list): A list of section IDs to extract links from.

        Returns:
        list: A list of full URLs without fragments.
        """
        if self.soup is None:
            raise ValueError("Content not fetched. Call fetch_content() first.")

        links = []
        for section_id in section_ids:
            links.extend(self.extract_links_from_section(section_id))
        return links
