import streamlit as st
from streamlit_option_menu import option_menu

from viz_model import VizModel, VizSituation, index_to_widget

import overtourism as ot

edit_values = {}

def on_change():
    st.session_state.edit_model = st.session_state.app.add_model(st.session_state.context_name, edit_values, save=False)

def init_parameters(app):
    res = {}
    for group in app.groups:
        with st.expander(group['label'], False):
            if 'parameters' in group:
                for p in group['parameters']:  
                    res[p.name] = index_to_widget(st, p, on_change=on_change)
    return res


@st.dialog("Add scenario")
def add_scenario_dialog(app):
    c = init_parameters(app)
    context_name = st.text_input("Scenario name")
    
    if st.button("Add", type="primary", disabled=context_name == ""):
        if context_name != '':
            app.add_model(context_name, c)
            st.rerun()

#################################################################################
#####################          STREAMLIT APP              #######################
#################################################################################
st.set_page_config(
    page_title="Overtourism",
    layout="wide",
    initial_sidebar_state="expanded"
)


with st.sidebar:
    selected = option_menu(
        menu_title = "Overtourism",
        options = ["Home"],
        icons = ["house"],
        menu_icon = "cast",
        default_index = 0,
    )

if selected == "Home":

    if "app" not in st.session_state:
        st.session_state.app = ot.app()
        st.session_state.editing = False
        
    # Create a row layout
    c_main, c_form = st.columns([2,1], vertical_alignment="bottom")

    # Parameters
    with c_form:
        pass

    # Charts and legends
    with c_main:
        context_name = st.selectbox(
            "Situation",
            map(lambda x: x.name, st.session_state.app.situations),
        )

    for i, mv in enumerate(st.session_state.app.vis_models):
        c_plot, c_data = st.columns([2,2])
        with c_plot:
            st.pyplot(mv.viz(context_name), use_container_width=False, clear_figure=True)
        with c_data:
            if mv.name != "base model":
                if st.button("Remove", type="primary", key=mv.name + "_remove", icon=":material/delete:"):
                    st.session_state.app.remove_model(mv.name)
                    st.rerun()
            if mv.change:
                st.subheader("Changes")
                for i in mv.change:
                    st.write(f"{i.name} : {i}")

            if mv.kpis and mv.kpis[context_name]:
                st.subheader("KPIs")
                for key in mv.kpis[context_name]:
                    st.write(f"{key} - {mv.kpis[context_name][key]}")

    if st.session_state.editing and st.session_state.edit_model:
        c_temp_model, c_temp_form = st.columns([2,2])
        with c_temp_model:
            st.pyplot(st.session_state.edit_model.viz(context_name), use_container_width=False, clear_figure=True)
            
        with c_temp_form:
            edit_values = init_parameters(st.session_state.app)
            context_name = st.text_input("Scenario name", value = st.session_state.edit_model.name, key = "context_name")
    
            if st.button("Add", type="primary", disabled=context_name == ""):
                if context_name != '':
                    st.session_state.edititing = False
                    st.session_state.edit_model = None
                    st.session_state.app.add_model(context_name, edit_values)
                    edit_values = {}
                    st.rerun()
    
    else:
        if st.button("Add scenario", type="primary"):
            st.session_state.editing = True
            st.session_state.edit_model = st.session_state.app.add_model(f"Scenario {len(st.session_state.app.vis_models)}", save=False)
            st.rerun()
            #add_scenario_dialog(st.session_state.app)
                    
        # # Lengend
        # st.subheader("Legenda")
        # for cat_obj in configuration['categories']:
        #     if 'constraints' in cat_obj:
        #         for p in cat_obj['constraints']:
        #             if 'label' in p:
        #                 st.caption(p['label'])
        #             if 'description' in p:
        #                 st.markdown(p['description'])
        #             # st.divider()
