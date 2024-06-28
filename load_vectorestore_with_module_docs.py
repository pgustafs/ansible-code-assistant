from ansible_doc import LinkExtractor, AnsibleDocLoader

url = "https://docs.ansible.com/ansible/latest/collections/index_module.html#ansible-builtin"
base_url = "https://docs.ansible.com/ansible/latest/collections/"

extractor = LinkExtractor(url, base_url)
extractor.fetch_content()
section_ids = ['ansible-builtin', 'community-libvirt']
links = extractor.extract_links_from_sections(section_ids)

# Display the list of links
for link in links:
    print(link)