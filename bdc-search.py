import webbrowser

import pystac_client

# Criar cliente do STAC-INPE
service = pystac_client.Client.open("https://data.inpe.br/bdc/stac/v1/")

# Definir bounding box para busca
bbox_sinop = [-55.62, -11.95, -55.40, -11.78]

# Criar pesquisa de coleções
item_search = service.search(
    collections=["CB4A-WPM-L4-DN-1"],
    bbox=bbox_sinop,
    datetime="2026-07-01/2026-08-10",
    query={"eo:cloud_cover": {"lt": 10}},
)

# Exibir qtd itens encontrados
itens = list(item_search.items())
print(f"Encontrados: {len(itens)} itens\n")

# Exibir detalhes dos itens encontrados
for idx, item in enumerate(itens):
    date = item.properties.get("datetime", "?")
    print(f"[{idx}] {item.id} | data: {date}")

# Abrir thumbnail do item selecionado
choice = input("\nDigite o índice para abrir thumbnail (ou Enter para sair): ")
if choice.strip():
    item = itens[int(choice)]
    url = item.assets["thumbnail"].href
    print(f"Abrindo: {url}")
    webbrowser.open(url)
